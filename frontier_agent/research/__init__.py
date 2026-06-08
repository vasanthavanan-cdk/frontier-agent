"""Research subsystem: web search, content fetching, and synthesis.

Pipeline: generate_queries → search → fetch → synthesize → research_context
The output (research_context) is injected into AgentState before the planner
runs, giving all downstream nodes access to current, sourced information.
"""
