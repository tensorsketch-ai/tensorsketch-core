"""Interop — bridges between TensorSketch and the wider agent ecosystem.

Protocol adapters live here, each an optional install imported only when you use it, so the core
stays dependency-free. Today: **MCP** (Model Context Protocol) in `tensorsketch.interop.mcp` —
consume
external tool servers as TensorSketch tools, and expose TensorSketch tools to any MCP client. OTel
tracing, A2A,
and AG-UI land here in later increments.
"""
