from lifeos.agents import LLMResponse, StubLLM, ToolCall, answer_agent, capture_agent, planner_agent, run_agent
from lifeos.agents.tools import memory_tools


def test_capture_agent_stores_and_categorises(memory):
    run = capture_agent(memory, StubLLM()).run("Jonas suggested a case for Garden Bot. #electronics")
    assert [t["tool"] for t in run.trace] == ["add_note"]
    assert "stored" in run.final_text and "Person: Jonas" in run.final_text
    again = capture_agent(memory, StubLLM()).run("Jonas suggested a case for Garden Bot. #electronics")
    assert "already stored" in again.final_text


def test_answer_agent_cites_provenance(loaded):
    run = answer_agent(loaded, StubLLM()).run("What did Lena recommend?")
    assert run.trace[0]["tool"] == "search_memory" and run.steps == 2
    assert "why:" in run.final_text


def test_planner_agent_reads_tasks_from_graph(loaded):
    run = planner_agent(loaded, StubLLM()).run("Plan the Lisbon Trip")
    assert [t["tool"] for t in run.trace] == ["search_memory", "get_entity"]
    assert "Book flights" in run.final_text and "owner: Maya" in run.final_text


def test_loop_reports_tool_errors_and_unknown_tools(loaded):
    seen = []

    class Scripted:
        def generate(self, system, messages, tools):
            seen.append(messages[-1])
            if len(seen) == 1:
                return LLMResponse(tool_calls=[ToolCall("1", "get_entity", {"name": "nobody"}),
                                               ToolCall("2", "missing_tool", {})])
            return LLMResponse(text="done")

    run = run_agent(Scripted(), "sys", "hi", [memory_tools(loaded)["get_entity"]])
    assert run.final_text == "done"
    results = seen[1]["content"]
    assert all(r["is_error"] for r in results)
    assert [t["error"] for t in run.trace][1].startswith("unknown tool")


def test_loop_stops_at_max_steps(loaded):
    class Forever:
        def generate(self, system, messages, tools):
            return LLMResponse(tool_calls=[ToolCall(str(len(messages)), "search_memory", {"query": "x"})])

    run = run_agent(Forever(), "sys", "hi", [memory_tools(loaded)["search_memory"]], max_steps=3)
    assert run.steps == 3 and "stopped after 3 steps" in run.final_text
