import asyncio
import os
from typing import Any
import dotenv
from github import Github
from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentWorkflow,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
dotenv.load_dotenv()
llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_BASE_URL"),
)
git = Github(os.getenv("GITHUB_TOKEN")) if os.getenv("GITHUB_TOKEN") else None
repository = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")
repo = None
if git is not None and repository:
    repo = git.get_repo(repository)
CONTEXT_SYSTEM_PROMPT = """You are the context gathering agent. When gathering context, you MUST gather
: 
  - The details: author, title, body, diff_url, state, and head_sha; 
  - Changed files; 
  - Any requested for files; 
Once you gather the requested info, save it with add_gathered_contexts_to_state, then hand off to CommentorAgent."""
COMMENTOR_SYSTEM_PROMPT = """You are the commentor agent that writes review comments for pull requests as a human reviewer would.
Ensure to do the following for a thorough review:
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent.
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing:
    - What is good about the PR?
    - Did the author follow ALL contribution rules? What is missing?
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.
    - Are new endpoints documented? - use the diff to determine this.
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.
 - If you need any additional details, you must hand off to the ContextAgent.
 - You must hand off to the ReviewAndPostingAgent once you are done drafting a review.
 - You should directly address the author. So your comments should sound like:
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?" """
REVIEW_POSTING_SYSTEM_PROMPT = """You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: 
   - Be a ~200-300 word review in markdown format. 
   - Specify what is good about the PR: 
   - Did the author follow ALL contribution rules? What is missing? 
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? 
   - Are there notes on whether new endpoints were documented? 
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? 
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. 
 When you are satisfied, post the review to GitHub.  
 When you are satisfied, you MUST call the post_pr_review tool with pr_number=<the PR number from the user message> and comment set to the final review text. Do not finish without calling post_pr_review.
"""

def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Use this tool to fetch pull request details for a given PR number (author, title, body, state, commits)."""
    pull_request = repo.get_pull(pr_number)
    commit_shas: list[str] = []
    for commit in pull_request.get_commits():
        commit_shas.append(commit.sha)
    body = pull_request.body
    if not body:
        body = "No description provided."
    return {
        "user": pull_request.user.login,
        "title": pull_request.title,
        "body": body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "commit_SHAs": commit_shas,
        "head_sha": pull_request.head.sha,
    }
def get_file_contents(file_path: str) -> str:
    """Use this tool to fetch the full text contents of a file from the repository given its path."""
    file_obj = repo.get_contents(file_path)
    return file_obj.decoded_content.decode("utf-8")
def get_pr_commit_details(commit_sha: str) -> list[dict[str, Any]]:
    """Use this tool to fetch changed files and diffs for a commit given its SHA (from PR details head_sha)."""
    commit = repo.get_commit(commit_sha)
    changed_files: list[dict[str, Any]] = []
    for changed in commit.files:
        changed_files.append(
            {
                "filename": changed.filename,
                "status": changed.status,
                "additions": changed.additions,
                "deletions": changed.deletions,
                "changes": changed.changes,
                "patch": changed.patch,
            }
        )
    return changed_files
async def add_gathered_contexts_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Useful for saving gathered repository context into workflow state for the CommentorAgent."""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_contexts
    await ctx.store.set("state", current_state)
    return "State updated with gathered contexts."
async def add_draft_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for saving the draft PR review comment into workflow state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "State updated with draft comment."
async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for saving the approved final PR review into workflow state."""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "State updated with final review."
def post_pr_review(pr_number: int, comment: str) -> str:
    """Post a pull request review to GitHub with the given comment body for the PR number."""
    pull_request = repo.get_pull(int(pr_number))
    pull_request.create_review(body=comment, event="COMMENT")
    return "Review posted successfully."
pr_details_tool = FunctionTool.from_defaults(get_pr_details)
file_contents_tool = FunctionTool.from_defaults(get_file_contents)
pr_commit_details_tool = FunctionTool.from_defaults(get_pr_commit_details)
post_pr_review_tool = FunctionTool.from_defaults(post_pr_review)
context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers pull request and repository context from GitHub for review.",
    tools=[
        pr_details_tool,
        file_contents_tool,
        pr_commit_details_tool,
        add_gathered_contexts_to_state,
    ],
    system_prompt=CONTEXT_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"],
)
commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[add_draft_comment_to_state],
    system_prompt=COMMENTOR_SYSTEM_PROMPT,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)
review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Review the generated draft comment, check if refinements are needed, and finally post it to GitHub.",
    tools=[add_final_review_to_state, post_pr_review_tool],
    system_prompt=REVIEW_POSTING_SYSTEM_PROMPT,
    can_handoff_to=["CommentorAgent"],
)
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)
async def main() -> None:
    query = f"Write a review for PR number {pr_number}"
    prompt = RichPromptTemplate(query)
    handler = workflow_agent.run(prompt.format())
    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

    await handler
if __name__ == "__main__":
    asyncio.run(main())
    if git is not None:
        git.close()
