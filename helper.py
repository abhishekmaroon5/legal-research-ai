import warnings
warnings.filterwarnings("ignore", message=".*TqdmWarning.*")
from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel
from tavily import TavilyClient
import os
import sqlite3

class AgentState(TypedDict):
    legal_question: str
    research_plan: str
    sources: List[str]
    case_summaries: List[str]
    argument_draft: str
    critique: str
    revision_number: int
    max_revisions: int
    count: Annotated[int, operator.add]
    lnode: str


class Queries(BaseModel):
    queries: List[str]
    
class ewriter():
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.PLAN_PROMPT = """You are an experienced law librarian. Given the student's legal question, create a detailed research plan that includes:
1. Key legal concepts to research
2. Types of sources to consult (cases, statutes, regulations, secondary sources)
3. Suggested search strategies
4. Important jurisdictions to consider
5. Potential counterarguments to research

Provide a structured outline that will guide the research process."""
        self.WRITER_PROMPT = """You are a law student writing a legal memorandum. Using the research plan, sources, and case summaries, draft a comprehensive legal argument that:
1. States the legal question
2. Presents relevant rules and precedents
3. Applies the law to the facts
4. Addresses potential counterarguments
5. Reaches a conclusion
6. Includes proper citations

Ensure your argument is clear, logical, and well-supported by authority."""
        self.RESEARCH_PLAN_PROMPT = """You are a legal research assistant. Based on the research plan and legal question, generate 3-5 specific search queries to find relevant:
- Case law
- Statutes
- Regulations
- Law review articles
- Secondary sources

Format each query to maximize relevant results."""
        self.REFLECTION_PROMPT = """You are a law professor reviewing a legal memorandum. Provide detailed feedback on:
1. Legal analysis and reasoning
2. Use of authority and citations
3. Structure and organization
4. Clarity and precision
5. Counterargument analysis
6. Areas for improvement

Be specific and constructive in your critique."""
        self.RESEARCH_CRITIQUE_PROMPT = """You are a legal research assistant. Based on the critique and legal question, generate 2-3 specific search queries to find additional relevant:
- Case law
- Statutes
- Regulations
- Law review articles
- Secondary sources

Focus on addressing the gaps identified in the critique."""
        self.tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.plan_node)
        builder.add_node("research_plan_node", self.research_plan_node)
        builder.add_node("generate", self.generation_node)
        builder.add_node("critique_node", self.reflection_node)
        builder.add_node("research_critique", self.research_critique_node)
        builder.set_entry_point("planner")
        builder.add_conditional_edges(
            "generate", 
            self.should_continue, 
            {END: END, "critique_node": "critique_node"}
        )
        builder.add_edge("planner", "research_plan_node")
        builder.add_edge("research_plan_node", "generate")
        builder.add_edge("critique_node", "research_critique")
        builder.add_edge("research_critique", "generate")
        memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_after=['planner', 'generate', 'critique_node', 'research_plan_node', 'research_critique']
        )


    def plan_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.PLAN_PROMPT), 
            HumanMessage(content=state['legal_question'])
        ]
        response = self.model.invoke(messages)
        return {
            "research_plan": response.content,
            "lnode": "planner",
            "count": 1
        }
    def research_plan_node(self, state: AgentState):
        try:
            queries = self.model.with_structured_output(Queries).invoke([
                SystemMessage(content=self.RESEARCH_PLAN_PROMPT),
                HumanMessage(content=f"Question: {state['legal_question']}\nPlan: {state['research_plan']}")
            ])
            sources = state['sources'] or []
            
            # Try to get sources from Tavily
            try:
                for q in queries.queries:
                    try:
                        response = self.tavily.search(query=q, max_results=2)
                        for r in response['results']:
                            sources.append(r['content'])
                    except Exception as e:
                        # If Tavily search fails, add a placeholder
                        sources.append(f"Search failed for query: {q}. Error: {str(e)}")
                        print(f"Tavily search error: {str(e)}")
            except Exception as e:
                print(f"Tavily API error: {str(e)}")
                # Add a fallback source if all searches fail
                sources.append("Unable to fetch sources from Tavily. Please check your API key and internet connection.")
            
            return {
                "sources": sources,
                "queries": queries.queries,
                "lnode": "research_plan_node",
                "count": 1
            }
        except Exception as e:
            print(f"Error in research_plan_node: {str(e)}")
            return {
                "sources": ["Error occurred during research planning. Please try again."],
                "queries": [],
                "lnode": "research_plan_node",
                "count": 1
            }
    def generation_node(self, state: AgentState):
        content = "\n\n".join(state['sources'] or [])
        user_message = HumanMessage(
            content=f"Question: {state['legal_question']}\nPlan: {state['research_plan']}\n\n{content}")
        messages = [
            SystemMessage(content=self.WRITER_PROMPT),
            user_message
        ]
        response = self.model.invoke(messages)
        return {
            "argument_draft": response.content,
            "revision_number": state.get("revision_number", 1) + 1,
            "lnode": "generate",
            "count": 1
        }
    def reflection_node(self, state: AgentState):
        messages = [
            SystemMessage(content=self.REFLECTION_PROMPT),
            HumanMessage(content=state['argument_draft'])
        ]
        response = self.model.invoke(messages)
        return {
            "critique": response.content,
            "lnode": "critique_node",
            "count": 1
        }
    def research_critique_node(self, state: AgentState):
        try:
            queries = self.model.with_structured_output(Queries).invoke([
                SystemMessage(content=self.RESEARCH_CRITIQUE_PROMPT),
                HumanMessage(content=state['critique'])
            ])
            sources = state['sources'] or []
            
            # Try to get sources from Tavily
            try:
                for q in queries.queries:
                    try:
                        response = self.tavily.search(query=q, max_results=2)
                        for r in response['results']:
                            sources.append(r['content'])
                    except Exception as e:
                        # If Tavily search fails, add a placeholder
                        sources.append(f"Search failed for query: {q}. Error: {str(e)}")
                        print(f"Tavily search error: {str(e)}")
            except Exception as e:
                print(f"Tavily API error: {str(e)}")
                # Add a fallback source if all searches fail
                sources.append("Unable to fetch sources from Tavily. Please check your API key and internet connection.")
            
            return {
                "sources": sources,
                "lnode": "research_critique",
                "count": 1
            }
        except Exception as e:
            print(f"Error in research_critique_node: {str(e)}")
            return {
                "sources": ["Error occurred during research critique. Please try again."],
                "lnode": "research_critique",
                "count": 1
            }
    def should_continue(self, state):
        if state["revision_number"] > state["max_revisions"]:
            return END
        return "critique_node"

import gradio as gr
import time

class writer_gui( ):
    def __init__(self, graph, share=False):
        self.graph = graph
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = 10
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.demo = self.create_interface()

    def updt_disp(self, topic_bx, lnode_bx, nnode_bx, threadid_bx, revision_bx, count_bx, step_pd, thread_pd):
        ''' general update display on state change '''
        current_state = self.graph.get_state(self.thread)
        hist = []
        for state in self.graph.get_state_history(self.thread):
            if state.metadata['step'] < 1:
                continue
            s_thread_ts = state.config['configurable']['thread_ts']
            s_tid = state.config['configurable']['thread_id']
            s_count = state.values.get('count', 0)
            s_lnode = state.values.get('lnode', '')
            s_rev = state.values.get('revision_number', 0)
            s_nnode = state.next
            st = f"{s_tid}:{s_count}:{s_lnode}:{s_nnode}:{s_rev}:{s_thread_ts}"
            hist.append(st)
        if not current_state.metadata:
            return {}
        else:
            return {
                topic_bx: current_state.values.get("legal_question", ""),
                lnode_bx: current_state.values.get("lnode", ""),
                count_bx: current_state.values.get("count", 0),
                revision_bx: current_state.values.get("revision_number", 0),
                nnode_bx: current_state.next,
                threadid_bx: self.thread_id,
                thread_pd: gr.Dropdown(label="Select Thread", choices=self.threads, value=self.thread_id, interactive=True),
                step_pd: gr.Dropdown(label="Select Step", choices=hist, value=hist[0] if hist else None, interactive=True),
            }

    def get_snapshots(self):
        new_label = f"thread_id: {self.thread_id}, Summary of snapshots"
        sstate = ""
        for state in self.graph.get_state_history(self.thread):
            for key in ['research_plan', 'argument_draft', 'critique']:
                if key in state.values:
                    state.values[key] = state.values[key][:80] + "..."
            if 'sources' in state.values:
                for i in range(len(state.values['sources'])):
                    state.values['sources'][i] = state.values['sources'][i][:20] + '...'
            if 'writes' in state.metadata:
                state.metadata['writes'] = "not shown"
            sstate += str(state) + "\n\n"
        return gr.update(label=new_label, value=sstate)

    def vary_btn(self, stat):
        return(gr.update(variant=stat))

    def run_agent(self, start, topic, stop_after):
        if start:
            self.iterations.append(0)
            config = {
                'legal_question': topic,
                "max_revisions": 2,
                "revision_number": 0,
                'lnode': "",
                "research_plan": "no plan",
                "argument_draft": "no draft",
                "critique": "no critique",
                "sources": ["no content"],
                "queries": "no queries",
                "count": 0
            }
            self.thread_id += 1
            self.threads.append(self.thread_id)
            self.partial_message = ""  # Reset message on start
        else:
            config = None
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        while self.iterations[self.thread_id] < self.max_iterations:
            self.response = self.graph.invoke(config, self.thread)
            self.iterations[self.thread_id] += 1
            
            # Format the response for better display
            current_state = self.graph.get_state(self.thread)
            current_node = current_state.values.get("lnode", "")
            next_node = current_state.next
            
            # Create a formatted message
            formatted_message = f"""
### Current Stage: {current_node}
**Next Stage:** {next_node}

**Progress Update:**
{self.format_state_output(current_state)}

---
"""
            self.partial_message += formatted_message
            
            # Update display state
            lnode, nnode, _, rev, acount = self.get_disp_state()
            
            # Return only the formatted message for the Markdown component
            yield self.partial_message
            
            config = None
            if not nnode:  
                return
            if lnode in stop_after:
                return

    def format_state_output(self, state):
        """Format the state output for better readability"""
        output = []
        
        # Format research plan
        if state.values.get("research_plan") and state.values["research_plan"] != "no plan":
            output.append("**Research Plan:**")
            output.append(state.values["research_plan"])
            output.append("")
        
        # Format sources
        if state.values.get("sources") and state.values["sources"] != ["no content"]:
            output.append("**Sources Found:**")
            for i, source in enumerate(state.values["sources"], 1):
                output.append(f"{i}. {source[:200]}...")
            output.append("")
        
        # Format argument draft
        if state.values.get("argument_draft") and state.values["argument_draft"] != "no draft":
            output.append("**Current Argument:**")
            output.append(state.values["argument_draft"])
            output.append("")
        
        # Format critique
        if state.values.get("critique") and state.values["critique"] != "no critique":
            output.append("**Expert Critique:**")
            output.append(state.values["critique"])
            output.append("")
        
        return "\n".join(output)

    def get_disp_state(self):
        current_state = self.graph.get_state(self.thread)
        lnode = current_state.values.get("lnode", "")
        acount = current_state.values.get("count", 0)
        rev = current_state.values.get("revision_number", 0)
        nnode = current_state.next
        return lnode, nnode, self.thread_id, rev, acount
    
    def get_state(self,key):
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            lnode,nnode,self.thread_id,rev,astep = self.get_disp_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return gr.update(label=new_label, value=current_values.values[key])
        else:
            return ""  
    
    def get_content(self,):
        current_values = self.graph.get_state(self.thread)
        if "sources" in current_values.values:
            content = current_values.values["sources"]
            lnode,nnode,thread_id,rev,astep = self.get_disp_state()
            new_label = f"last_node: {lnode}, thread_id: {self.thread_id}, rev: {rev}, step: {astep}"
            return gr.update(label=new_label, value="\n\n".join(item for item in content) + "\n\n")
        else:
            return ""  
    
    def update_hist_pd(self,):
        #print("update_hist_pd")
        hist = []
        # curiously, this generator returns the latest first
        for state in self.graph.get_state_history(self.thread):
            if state.metadata['step'] < 1:
                continue
            thread_ts = state.config['configurable']['thread_ts']
            tid = state.config['configurable']['thread_id']
            count = state.values['count']
            lnode = state.values['lnode']
            rev = state.values['revision_number']
            nnode = state.next
            st = f"{tid}:{count}:{lnode}:{nnode}:{rev}:{thread_ts}"
            hist.append(st)
        return gr.Dropdown(label="update_state from: thread:count:last_node:next_node:rev:thread_ts", 
                           choices=hist, value=hist[0],interactive=True)
    
    def find_config(self,thread_ts):
        for state in self.graph.get_state_history(self.thread):
            config = state.config
            if config['configurable']['thread_ts'] == thread_ts:
                return config
        return(None)
            
    def copy_state(self,hist_str):
        ''' result of selecting an old state from the step pulldown. Note does not change thread. 
             This copies an old state to a new current state. 
        '''
        thread_ts = hist_str.split(":")[-1]
        #print(f"copy_state from {thread_ts}")
        config = self.find_config(thread_ts)
        #print(config)
        state = self.graph.get_state(config)
        self.graph.update_state(self.thread, state.values, as_node=state.values['lnode'])
        new_state = self.graph.get_state(self.thread)  #should now match
        new_thread_ts = new_state.config['configurable']['thread_ts']
        tid = new_state.config['configurable']['thread_id']
        count = new_state.values['count']
        lnode = new_state.values['lnode']
        rev = new_state.values['revision_number']
        nnode = new_state.next
        return lnode,nnode,new_thread_ts,rev,count
    
    def update_thread_pd(self,):
        #print("update_thread_pd")
        return gr.Dropdown(label="choose thread", choices=threads, value=self.thread_id,interactive=True)
    
    def switch_thread(self,new_thread_id):
        #print(f"switch_thread{new_thread_id}")
        self.thread = {"configurable": {"thread_id": str(new_thread_id)}}
        self.thread_id = new_thread_id
        return 
    
    def modify_state(self,key,asnode,new_state):
        ''' gets the current state, modifes a single value in the state identified by key, and updates state with it.
        note that this will create a new 'current state' node. If you do this multiple times with different keys, it will create
        one for each update. Note also that it doesn't resume after the update
        '''
        current_values = self.graph.get_state(self.thread)
        current_values.values[key] = new_state
        self.graph.update_state(self.thread, current_values.values,as_node=asnode)
        return


    def create_interface(self):
        with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="blue",
                neutral_hue="slate",
                spacing_size="sm",
                radius_size="md",
                text_size="md"
            )
        ) as demo:
            gr.Markdown("""
            # Legal Research AI Assistant
            An AI-powered tool for legal research, case analysis, and argument drafting.
            """)
            
            with gr.Tab("Research Assistant"):
                with gr.Row():
                    with gr.Column(scale=2):
                        topic_bx = gr.Textbox(
                            label="Legal Question",
                            placeholder="Enter your legal question here...",
                            value="What is the standard for summary judgment in federal court?",
                            lines=3
                        )
                    with gr.Column(scale=1):
                        with gr.Row():
                            gen_btn = gr.Button("Start Research", variant="primary", size="lg")
                            cont_btn = gr.Button("Continue Research", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        lnode_bx = gr.Textbox(label="Current Stage", min_width=100)
                        nnode_bx = gr.Textbox(label="Next Stage", min_width=100)
                    with gr.Column(scale=1):
                        threadid_bx = gr.Textbox(label="Thread ID", scale=0, min_width=80)
                        revision_bx = gr.Textbox(label="Revision", scale=0, min_width=80)
                        count_bx = gr.Textbox(label="Step Count", scale=0, min_width=80)
                
                with gr.Accordion("Advanced Controls", open=False):
                    checks = list(self.graph.nodes.keys())
                    checks.remove('__start__')
                    stop_after = gr.CheckboxGroup(
                        checks,
                        label="Interrupt After Stage",
                        value=checks,
                        scale=0,
                        min_width=400
                    )
                    with gr.Row():
                        thread_pd = gr.Dropdown(
                            choices=self.threads,
                            interactive=True,
                            label="Select Thread",
                            min_width=120,
                            scale=0
                        )
                        step_pd = gr.Dropdown(
                            choices=['N/A'],
                            interactive=True,
                            label="Select Step",
                            min_width=160,
                            scale=1
                        )
                
                live = gr.Markdown(
                    label="Research Progress",
                    value="",
                    elem_classes=["research-progress"]
                )
            
            with gr.Tab("Research Plan"):
                with gr.Row():
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("Refresh Plan", variant="secondary")
                        modify_btn = gr.Button("Modify Plan", variant="primary")
                plan = gr.Textbox(
                    label="Research Plan",
                    lines=10,
                    interactive=True,
                    show_copy_button=True
                )
            
            with gr.Tab("Legal Sources"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Sources", variant="secondary")
                content_bx = gr.Textbox(
                    label="Research Sources",
                    lines=10,
                    show_copy_button=True
                )
            
            with gr.Tab("Legal Argument"):
                with gr.Row():
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("Refresh Argument", variant="secondary")
                        modify_btn = gr.Button("Modify Argument", variant="primary")
                draft_bx = gr.Textbox(
                    label="Legal Argument",
                    lines=10,
                    interactive=True,
                    show_copy_button=True
                )
            
            with gr.Tab("Expert Critique"):
                with gr.Row():
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("Refresh Critique", variant="secondary")
                        modify_btn = gr.Button("Modify Critique", variant="primary")
                critique_bx = gr.Textbox(
                    label="Expert Critique",
                    lines=10,
                    interactive=True,
                    show_copy_button=True
                )
            
            with gr.Tab("State History"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh History", variant="secondary")
                snapshots = gr.Textbox(
                    label="State History",
                    lines=10,
                    show_copy_button=True
                )
            
            # Add error message display
            error_msg = gr.Markdown(
                value="",
                visible=False,
                elem_classes=["error-message"]
            )
            
            # Add custom CSS for error messages
            gr.HTML("""
            <style>
            .error-message {
                background-color: #fff3f3;
                border: 1px solid #ffcdd2;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                color: #d32f2f;
            }
            </style>
            """)
            
            # Actions
            sdisps = [topic_bx, lnode_bx, nnode_bx, threadid_bx, revision_bx, count_bx, step_pd, thread_pd]
            
            thread_pd.input(self.switch_thread, [thread_pd], None).then(
                fn=lambda: self.updt_disp(*sdisps), inputs=None, outputs=sdisps
            )
            
            step_pd.input(self.copy_state, [step_pd], None).then(
                fn=lambda: self.updt_disp(*sdisps), inputs=None, outputs=sdisps
            )
            
            gen_btn.click(
                self.vary_btn, gr.Number("secondary", visible=False), gen_btn
            ).then(
                fn=self.run_agent,
                inputs=[gr.Number(True, visible=False), topic_bx, stop_after],
                outputs=[live],
                show_progress=True
            ).then(
                fn=lambda: self.updt_disp(*sdisps), inputs=None, outputs=sdisps
            ).then(
                self.vary_btn, gr.Number("primary", visible=False), gen_btn
            ).then(
                self.vary_btn, gr.Number("primary", visible=False), cont_btn
            )
            
            cont_btn.click(
                self.vary_btn, gr.Number("secondary", visible=False), cont_btn
            ).then(
                fn=self.run_agent,
                inputs=[gr.Number(False, visible=False), topic_bx, stop_after],
                outputs=[live],
                show_progress=True
            ).then(
                fn=lambda: self.updt_disp(*sdisps), inputs=None, outputs=sdisps
            ).then(
                self.vary_btn, gr.Number("primary", visible=False), cont_btn
            )
            
            # Plan tab actions
            with gr.Tab("Research Plan"):
                refresh_btn.click(
                    fn=self.get_state,
                    inputs=gr.Number("research_plan", visible=False),
                    outputs=plan
                )
                modify_btn.click(
                    fn=self.modify_state,
                    inputs=[gr.Number("research_plan", visible=False),
                           gr.Number("planner", visible=False), plan],
                    outputs=None
                ).then(
                    fn=lambda: self.updt_disp(*sdisps), inputs=None, outputs=sdisps
                )
            
            # Sources tab actions
            with gr.Tab("Legal Sources"):
                refresh_btn.click(
                    fn=self.get_content,
                    inputs=None,
                    outputs=content_bx
                )
            
            # Argument tab actions
            with gr.Tab("Legal Argument"):
                refresh_btn.click(
                    fn=self.get_state,
                    inputs=gr.Number("argument_draft", visible=False),
                    outputs=draft_bx
                )
                modify_btn.click(
                    fn=self.modify_state,
                    inputs=[gr.Number("argument_draft", visible=False),
                           gr.Number("generate", visible=False), draft_bx],
                    outputs=None
                ).then(
                    fn=lambda: self.updt_disp(*sdisps), inputs=None, outputs=sdisps
                )
            
            # Critique tab actions
            with gr.Tab("Expert Critique"):
                refresh_btn.click(
                    fn=self.get_state,
                    inputs=gr.Number("critique", visible=False),
                    outputs=critique_bx
                )
                modify_btn.click(
                    fn=self.modify_state,
                    inputs=[gr.Number("critique", visible=False),
                           gr.Number("critique_node", visible=False), critique_bx],
                    outputs=None
                ).then(
                    fn=lambda: self.updt_disp(*sdisps), inputs=None, outputs=sdisps
                )
            
            # History tab actions
            with gr.Tab("State History"):
                refresh_btn.click(
                    fn=self.get_snapshots,
                    inputs=None,
                    outputs=snapshots
                )
            
            # Add custom CSS for better formatting
            gr.HTML("""
            <style>
            .research-progress {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                border: 1px solid #e9ecef;
            }
            .research-progress h3 {
                color: #2c3e50;
                margin-bottom: 10px;
            }
            .research-progress strong {
                color: #3498db;
            }
            .research-progress hr {
                border: none;
                border-top: 1px solid #e9ecef;
                margin: 15px 0;
            }
            </style>
            """)
        
        return demo

    def launch(self, share=None):
        if port := os.getenv("PORT1"):
            try:
                self.demo.launch(share=True, server_port=int(port), server_name="0.0.0.0")
            except OSError:
                # If specified port is not available, try alternative ports
                for alt_port in range(8081, 8090):
                    try:
                        self.demo.launch(share=True, server_port=alt_port, server_name="0.0.0.0")
                        break
                    except OSError:
                        continue
        else:
            try:
                self.demo.launch(share=self.share)
            except OSError:
                # If default port is not available, try alternative ports
                for alt_port in range(8081, 8090):
                    try:
                        self.demo.launch(share=self.share, server_port=alt_port)
                        break
                    except OSError:
                        continue
