"""OpenAI function-call specs for the search/open/find tool set."""

SEARCH_SPEC = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search a Wikipedia knowledge base and return the top passages for a query. "
            "Use for multi-hop factual questions: search for one fact, read the passages "
            "(open a result to read it in full), then search again for the next hop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query (one hop / one fact at a time)."},
                "topn": {"type": "integer", "description": "Number of results to return.", "default": 10},
            },
            "required": ["query"],
        },
    },
}

OPEN_SPEC = {
    "type": "function",
    "function": {
        "name": "open",
        "description": (
            "Open a result from the MOST RECENT search by its numeric id (the 【id†...】 "
            "reference shown in the search results) to read its full text."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The numeric result id to open, from a preceding search."},
            },
            "required": ["id"],
        },
    },
}

FIND_SPEC = {
    "type": "function",
    "function": {
        "name": "find",
        "description": "Find a pattern (case-insensitive substring) in the currently open page.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text to search for in the current page."},
            },
            "required": ["pattern"],
        },
    },
}

search_specs = [SEARCH_SPEC, OPEN_SPEC, FIND_SPEC]
