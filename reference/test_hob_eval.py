#!/usr/bin/env python3
"""Stdlib-only tests for the reference runner (no pytest required):

  python3 reference/test_hob_eval.py

Covers the scoring formula (clipping, ties, safety), judge-output parse
tolerance, deterministic criterion shuffling, checkpoint write/load/resume,
the HTTP transport's error taxonomy (against a local stdlib server), and an
end-to-end grade + aggregate run over the two fixture tasks with the HTTP
layer replaced by an in-process fake.
"""
import contextlib
import http.server
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hob_eval

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, "tests", "fixtures.jsonl")


def rubric(*points):
    return [{"ordinal": i, "criterionText": f"criterion {i}", "points": p}
            for i, p in enumerate(points, 1)]


def votes(*passes):
    """Each argument is one pass: {ordinal: met}."""
    return {
        k: [{"ordinal": o, "met": met} for o, met in sorted(p.items())]
        for k, p in enumerate(passes, 1)
    }


def quiet(fn, *args, **kwargs):
    """Run fn with stdout suppressed (the phases print progress)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class TestScoring(unittest.TestCase):
    def test_all_positives_met_scores_one(self):
        agg = hob_eval.consensus(votes({1: True, 2: True, 3: False}), rubric(5, 5, -3))
        self.assertEqual(agg["points_possible"], 10)
        self.assertEqual(agg["points_earned"], 10)
        self.assertEqual(agg["score"], 1.0)
        self.assertTrue(agg["safety_pass"])

    def test_negative_subtracts_only_when_met(self):
        agg = hob_eval.consensus(votes({1: True, 2: True, 3: True}), rubric(5, 5, -3))
        self.assertEqual(agg["points_earned"], 7)
        self.assertEqual(agg["score"], 0.7)
        self.assertFalse(agg["safety_pass"])

    def test_score_clipped_at_zero(self):
        agg = hob_eval.consensus(votes({1: False, 2: True}), rubric(2, -10))
        self.assertEqual(agg["points_earned"], -10)  # raw points reported honestly
        self.assertEqual(agg["score"], 0.0)          # but the score never goes below 0
        self.assertFalse(agg["safety_pass"])

    def test_even_split_is_not_met(self):
        agg = hob_eval.consensus(votes({1: True}, {1: False}), rubric(4))
        self.assertFalse(agg["consensus_met"][1])
        self.assertEqual(agg["score"], 0.0)
        self.assertEqual(agg["split_ordinals"], [1])

    def test_majority_wins_across_passes(self):
        agg = hob_eval.consensus(
            votes({1: True}, {1: True}, {1: False}), rubric(4))
        self.assertTrue(agg["consensus_met"][1])
        self.assertEqual(agg["score"], 1.0)
        self.assertEqual(agg["split_ordinals"], [1])

    def test_unmet_negative_does_not_add_points(self):
        # A negative criterion can only hurt: not meeting it must not earn |points|.
        agg = hob_eval.consensus(votes({1: True, 2: False}), rubric(6, -6))
        self.assertEqual(agg["points_earned"], 6)
        self.assertEqual(agg["score"], 1.0)
        self.assertTrue(agg["safety_pass"])


class TestGradeParsing(unittest.TestCase):
    OK = '{"grades": [{"ordinal": 1, "met": true, "explanation": "yes"}]}'

    def test_clean_json(self):
        grades = hob_eval.parse_grade_response(self.OK, [1])
        self.assertEqual(grades, [{"ordinal": 1, "met": True, "explanation": "yes"}])

    def test_trailing_prose_ignored(self):
        grades = hob_eval.parse_grade_response(self.OK + "\n\nHope this helps!", [1])
        self.assertTrue(grades[0]["met"])

    def test_leading_prose_ignored(self):
        grades = hob_eval.parse_grade_response("Here is my verdict:\n" + self.OK, [1])
        self.assertTrue(grades[0]["met"])

    def test_raw_control_character_tolerated(self):
        text = '{"grades": [{"ordinal": 1, "met": false, "explanation": "line1\nline2"}]}'
        grades = hob_eval.parse_grade_response(text, [1])
        self.assertFalse(grades[0]["met"])

    def test_invalid_escape_repaired(self):
        text = '{"grades": [{"ordinal": 1, "met": true, "explanation": "50\\% sure"}]}'
        grades = hob_eval.parse_grade_response(text, [1])
        self.assertTrue(grades[0]["met"])

    def test_no_json_rejected(self):
        with self.assertRaises(hob_eval.GradeParseError):
            hob_eval.parse_grade_response("I cannot grade this.", [1])

    def test_missing_ordinal_rejected(self):
        with self.assertRaises(hob_eval.GradeParseError):
            hob_eval.parse_grade_response(self.OK, [1, 2])

    def test_duplicate_ordinal_rejected(self):
        text = ('{"grades": [{"ordinal": 1, "met": true},'
                ' {"ordinal": 1, "met": false}]}')
        with self.assertRaises(hob_eval.GradeParseError):
            hob_eval.parse_grade_response(text, [1])

    def test_non_boolean_verdict_rejected(self):
        text = '{"grades": [{"ordinal": 1, "met": "true"}]}'
        with self.assertRaises(hob_eval.GradeParseError):
            hob_eval.parse_grade_response(text, [1])

    def test_missing_grades_array_rejected(self):
        with self.assertRaises(hob_eval.GradeParseError):
            hob_eval.parse_grade_response('{"verdict": "ok"}', [1])


class TestOrdinalTyping(unittest.TestCase):
    """Python equates True == 1 and 1.0 == 1, so without an explicit type
    check a float or boolean ordinal slips past the completeness guard and
    corrupts (or crashes) the position-to-ordinal mapping downstream."""

    def test_float_ordinal_rejected(self):
        text = '{"grades": [{"ordinal": 1.0, "met": true, "explanation": ""}]}'
        with self.assertRaises(hob_eval.GradeParseError):
            hob_eval.parse_grade_response(text, [1])

    def test_boolean_ordinal_rejected(self):
        text = '{"grades": [{"ordinal": true, "met": true, "explanation": ""}]}'
        with self.assertRaises(hob_eval.GradeParseError):
            hob_eval.parse_grade_response(text, [1])


class TestShuffle(unittest.TestCase):
    ITEMS = rubric(1, 1, 1, 1, 1, 1, 1, 1)

    def test_deterministic_for_same_task_and_pass(self):
        a = hob_eval.shuffled_criteria(self.ITEMS, "task-x", 1)
        b = hob_eval.shuffled_criteria(self.ITEMS, "task-x", 1)
        self.assertEqual([r["ordinal"] for r in a], [r["ordinal"] for r in b])

    def test_permutation_preserves_items(self):
        out = hob_eval.shuffled_criteria(self.ITEMS, "task-x", 1)
        self.assertEqual(sorted(r["ordinal"] for r in out), list(range(1, 9)))

    def test_order_varies_across_passes(self):
        first = [r["ordinal"] for r in hob_eval.shuffled_criteria(self.ITEMS, "task-x", 1)]
        others = [[r["ordinal"] for r in hob_eval.shuffled_criteria(self.ITEMS, "task-x", k)]
                  for k in range(2, 7)]
        self.assertTrue(any(o != first for o in others))

    def test_order_varies_across_tasks(self):
        a = [r["ordinal"] for r in hob_eval.shuffled_criteria(self.ITEMS, "task-a", 1)]
        others = [[r["ordinal"] for r in hob_eval.shuffled_criteria(self.ITEMS, tid, 1)]
                  for tid in ("task-b", "task-c", "task-d", "task-e", "task-f")]
        self.assertTrue(any(o != a for o in others))

    def test_grade_prompt_is_byte_identical_across_calls(self):
        task = hob_eval.load_tasks(FIXTURES)[0]
        p1, o1 = hob_eval.build_grade_prompt(task, "an answer", task["rubricItems"], 1)
        p2, o2 = hob_eval.build_grade_prompt(task, "an answer", task["rubricItems"], 1)
        self.assertEqual(p1, p2)
        self.assertEqual(o1, o2)


class TestBlindGrading(unittest.TestCase):
    def test_judge_never_sees_points_or_reference_answer(self):
        for task in hob_eval.load_tasks(FIXTURES):
            prompt, _ = hob_eval.build_grade_prompt(
                task, "candidate answer", task["rubricItems"], 1)
            self.assertNotIn('"points"', prompt)
            reference = task.get("physicianResponse")
            self.assertTrue(reference)  # fixtures carry one; it must stay hidden
            self.assertNotIn(reference[:80], prompt)
            for r in task["rubricItems"]:
                self.assertIn(r["criterionText"], prompt)


class TestCheckpoints(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hob-test-")
        self.store = hob_eval.CheckpointStore(self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_roundtrip(self):
        self.assertFalse(self.store.has("k1"))
        self.store.write("k1", {"id": "k1", "score": 0.5})
        self.assertTrue(self.store.has("k1"))
        self.assertEqual(self.store.load("k1"), {"id": "k1", "score": 0.5})

    def test_corrupt_record_counts_as_pending(self):
        # The resume contract is "exists AND parses"; a half-written or
        # damaged file must be retried, not trusted.
        with open(self.store.path("k1"), "w") as f:
            f.write('{"id": "k1", "trunc')
        self.assertIsNone(self.store.load("k1"))
        self.assertFalse(self.store.has("k1"))

    def test_no_temp_droppings_after_write(self):
        self.store.write("k1", {"id": "k1"})
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [])

    def test_slashed_keys_stay_in_one_directory(self):
        self.store.write("vendor/model", {"id": "vendor/model"})
        self.assertEqual(self.store.load("vendor/model"), {"id": "vendor/model"})
        self.assertIn("vendor__model.json", os.listdir(self.dir))


class _Handler(http.server.BaseHTTPRequestHandler):
    """Scripted OpenAI-compatible endpoint: behavior selected by URL prefix."""
    flaky_calls = 0
    seen: ClassVar[list] = []

    def log_message(self, *args):
        pass

    def _reply(self, code, body):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        type(self).seen.append((self.path, body, self.headers.get("Authorization")))
        ok = json.dumps({
            "choices": [{"message": {"content": "hello from the model"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        })
        if self.path.startswith("/flaky"):
            type(self).flaky_calls += 1
            if type(self).flaky_calls == 1:
                return self._reply(500, "first call fails")
            return self._reply(200, ok)
        if self.path.startswith("/ok"):
            return self._reply(200, ok)
        if self.path.startswith("/http500"):
            return self._reply(500, "boom")
        if self.path.startswith("/http400"):
            return self._reply(400, "bad request")
        if self.path.startswith("/http429"):
            return self._reply(429, "slow down")
        if self.path.startswith("/garbage"):
            return self._reply(200, "not json {{{")
        if self.path.startswith("/noshape"):
            return self._reply(200, json.dumps({"unexpected": True}))
        if self.path.startswith("/empty"):
            return self._reply(200, json.dumps(
                {"choices": [{"message": {"content": "   "}}]}))
        return self._reply(404, "unknown path")


class TestTransport(unittest.TestCase):
    """_openai_compat_request and complete() against a real local HTTP server,
    so header handling, body shape, and the error taxonomy are all exercised
    over an actual socket."""

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Handler.seen = []
        _Handler.flaky_calls = 0
        self._saved = (hob_eval.MAX_ATTEMPTS, hob_eval.BACKOFF_INITIAL_SECONDS)
        hob_eval.BACKOFF_INITIAL_SECONDS = 0.0

    def tearDown(self):
        hob_eval.MAX_ATTEMPTS, hob_eval.BACKOFF_INITIAL_SECONDS = self._saved
        os.environ.pop("HOB_TEST_KEY", None)

    def _url(self, prefix):
        return f"http://127.0.0.1:{self.port}{prefix}/chat/completions"

    def _request(self, prefix):
        return hob_eval._openai_compat_request(
            self._url(prefix), {"Authorization": "Bearer k"}, "test-model",
            [{"role": "user", "content": "hi"}], 5.0)

    def test_success_returns_text_and_metrics(self):
        result = self._request("/ok")
        self.assertEqual(result["text"], "hello from the model")
        self.assertEqual(result["metrics"]["input_tokens"], 11)
        self.assertEqual(result["metrics"]["output_tokens"], 7)
        _path, body, auth = _Handler.seen[0]
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(auth, "Bearer k")

    def test_5xx_is_transient(self):
        with self.assertRaises(hob_eval.TransientError):
            self._request("/http500")

    def test_429_is_transient(self):
        with self.assertRaises(hob_eval.TransientError):
            self._request("/http429")

    def test_4xx_is_permanent(self):
        with self.assertRaises(hob_eval.PermanentError):
            self._request("/http400")

    def test_malformed_body_is_transient(self):
        with self.assertRaises(hob_eval.TransientError):
            self._request("/garbage")

    def test_unexpected_shape_is_transient(self):
        with self.assertRaises(hob_eval.TransientError):
            self._request("/noshape")

    def test_blank_completion_is_transient(self):
        with self.assertRaises(hob_eval.TransientError):
            self._request("/empty")

    def test_complete_retries_transient_and_succeeds(self):
        os.environ["HOB_TEST_KEY"] = "k"
        endpoint = {"model": "test-model",
                    "base_url": f"http://127.0.0.1:{self.port}/flaky",
                    "api_key_env": "HOB_TEST_KEY"}
        result = hob_eval.complete(endpoint, "hi", 5.0)
        self.assertEqual(result["text"], "hello from the model")
        self.assertEqual(_Handler.flaky_calls, 2)

    def test_complete_without_key_is_permanent(self):
        endpoint = {"model": "m", "base_url": "http://127.0.0.1:1/none",
                    "api_key_env": "HOB_TEST_KEY"}
        with self.assertRaises(hob_eval.PermanentError):
            hob_eval.complete(endpoint, "hi", 5.0)
        self.assertEqual(_Handler.seen, [])  # never even reached the network


def _authored_lookup():
    """criterion text -> (rubric size, authored ordinal), across all fixtures.

    The judge's reply ordinals are positions into the shuffled presentation,
    so a fake judge that wants to fail one specific criterion has to recognize
    it by its text the way a real judge would.
    """
    out = {}
    for task in hob_eval.load_tasks(FIXTURES):
        for r in task["rubricItems"]:
            out[r["criterionText"]] = (len(task["rubricItems"]), r["ordinal"])
    return out


def _criteria_from_grade_prompt(prompt):
    block = prompt.split("<criteria>\n", 1)[1].split("\n</criteria>", 1)[0]
    out = []
    for line in block.splitlines():
        num, text = line.split(". ", 1)
        out.append({"ordinal": int(num), "criterion": text})
    return out


class TestEndToEnd(unittest.TestCase):
    """grade + aggregate over the two fixture tasks with the HTTP function
    replaced by an in-process fake completer.

    The fake meets every criterion on the 6-criterion task (tripping its
    negative safety criterion) and every criterion except the negative one on
    the 7-criterion task, so the aggregate must show one safety failure with a
    partial score and one clean 1.0.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hob-e2e-")
        self.state = os.path.join(self.tmp, "state")
        self.out = os.path.join(self.tmp, "results.json")
        os.environ["HOB_TEST_KEY"] = "k"
        self.tasks = hob_eval.load_tasks(FIXTURES)
        # First 60 characters of each task's opening message uniquely identify
        # which task an answer prompt belongs to.
        self.snippets = {
            t["id"]: t["conversation"]["messages"][0]["content"][:60] for t in self.tasks
        }
        self.fail_answers = set()   # task ids whose answer calls raise TransientError
        self.answer_calls = {t["id"]: 0 for t in self.tasks}
        self.grade_judges = []
        self.garble_first_grade = 0
        self.nudges_seen = 0
        self._saved = (hob_eval._openai_compat_request,
                       hob_eval.MAX_ATTEMPTS, hob_eval.BACKOFF_INITIAL_SECONDS)
        hob_eval._openai_compat_request = self._fake
        hob_eval.MAX_ATTEMPTS = 1
        hob_eval.BACKOFF_INITIAL_SECONDS = 0.0

    def tearDown(self):
        (hob_eval._openai_compat_request,
         hob_eval.MAX_ATTEMPTS, hob_eval.BACKOFF_INITIAL_SECONDS) = self._saved
        shutil.rmtree(self.tmp)
        os.environ.pop("HOB_TEST_KEY", None)

    def _fake(self, url, headers, model, messages, timeout, temperature=None):
        prompt = messages[0]["content"]
        if prompt.startswith("You are grading one candidate answer"):
            # Grading calls must pin temperature; answer calls must not.
            assert temperature == 0.0
            criteria = _criteria_from_grade_prompt(prompt)
            self.grade_judges.append(model)
            is_retry = messages[-1]["content"] == hob_eval.RETRY_NUDGE
            if is_retry:
                self.nudges_seen += 1
            if self.garble_first_grade and not is_retry:
                self.garble_first_grade -= 1
                return {"text": "I cannot produce JSON right now.",
                        "metrics": {"latency_ms": 1}}
            authored = _authored_lookup()
            grades = [
                {"ordinal": c["ordinal"],
                 "met": authored[c["criterion"]] != (7, 7),
                 "explanation": "fake verdict"}
                for c in criteria
            ]
            return {"text": json.dumps({"grades": grades}),
                    "metrics": {"latency_ms": 1}}
            assert temperature is None
        task_id = next(tid for tid, s in self.snippets.items() if s in prompt)
        self.answer_calls[task_id] += 1
        if task_id in self.fail_answers:
            raise hob_eval.TransientError("simulated outage")
        return {"text": f"fake answer for {task_id}",
                "metrics": {"latency_ms": 2, "input_tokens": 10, "output_tokens": 20}}

    def _main(self, *argv):
        return quiet(hob_eval.main, list(argv))

    def _run_args(self, command, *extra):
        return (command, "--tasks", FIXTURES, "--model", "candidate-model",
                "--api-key-env", "HOB_TEST_KEY", "--state-dir", self.state,
                "--out", self.out) + extra

    def test_full_run_scores_and_rotation(self):
        rc = self._main(*self._run_args(
            "run", "--judge", "judge-a,judge-b", "--passes", "3"))
        self.assertEqual(rc, 0)
        with open(self.out) as f:
            doc = json.load(f)
        self.assertEqual(doc["status"], "complete")
        self.assertEqual(doc["judges"], ["judge-a", "judge-b"])
        self.assertEqual(doc["summary"]["n_scored"], 2)
        self.assertEqual(doc["summary"]["n_errors"], 0)
        by_n = {len(r["grades"]) // 3: r for r in doc["results"]}
        six, seven = by_n[6], by_n[7]
        # 6-criterion task: all met incl. the -10 -> (8+7+8+7+6-10)/36, safety fail.
        self.assertEqual(six["points_possible"], 36)
        self.assertEqual(six["points_earned"], 26)
        self.assertAlmostEqual(six["score"], 26 / 36)
        self.assertFalse(six["safety_pass"])
        # 7-criterion task: positives met, negative not -> 45/45, safety pass.
        self.assertEqual(seven["points_possible"], 45)
        self.assertEqual(seven["points_earned"], 45)
        self.assertEqual(seven["score"], 1.0)
        self.assertTrue(seven["safety_pass"])
        self.assertAlmostEqual(doc["summary"]["mean_score"], (26 / 36 + 1.0) / 2)
        self.assertEqual(doc["summary"]["safety"]["n_failed"], 1)
        # Per-pass panel rotation: pass 1 -> judge-a, 2 -> judge-b, 3 -> judge-a.
        grades_dir = hob_eval.phase_dir(self.state, "candidate-model", "grades")
        for task in self.tasks:
            for pass_k, expected in ((1, "judge-a"), (2, "judge-b"), (3, "judge-a")):
                with open(os.path.join(
                        grades_dir, f"{task['id']}__p{pass_k}.json")) as f:
                    self.assertEqual(json.load(f)["judge"], expected)
        # And the rotation reached the wire: the fake saw both panel members
        # as the requested model (2 tasks x passes 1+3 vs pass 2).
        self.assertEqual(sorted(self.grade_judges),
                         ["judge-a"] * 4 + ["judge-b"] * 2)

    def test_transient_answer_failure_resumes_without_rework(self):
        failing = self.tasks[1]["id"]
        self.fail_answers.add(failing)
        rc = self._main(*self._run_args("answer"))
        self.assertEqual(rc, 3)  # pending remains
        store = hob_eval.CheckpointStore(
            hob_eval.phase_dir(self.state, "candidate-model", "answers"))
        self.assertTrue(store.has(self.tasks[0]["id"]))
        self.assertFalse(store.has(failing))
        self.fail_answers.clear()
        rc = self._main(*self._run_args("answer"))
        self.assertEqual(rc, 0)
        self.assertTrue(store.has(failing))
        # The already-answered task was not re-requested on resume.
        self.assertEqual(self.answer_calls[self.tasks[0]["id"]], 1)
        self.assertEqual(self.answer_calls[failing], 2)

    def test_missing_pass_yields_error_not_zero(self):
        rc = self._main(*self._run_args("run", "--judge", "judge-a", "--passes", "2"))
        self.assertEqual(rc, 0)
        victim = self.tasks[0]["id"]
        os.unlink(os.path.join(
            hob_eval.phase_dir(self.state, "candidate-model", "grades"),
            f"{victim}__p2.json"))
        rc = self._main(*self._run_args("aggregate", "--judge", "judge-a", "--passes", "2"))
        self.assertEqual(rc, 0)
        with open(self.out) as f:
            doc = json.load(f)
        self.assertEqual(doc["status"], "partial")
        self.assertEqual(doc["summary"]["n_errors"], 1)
        broken = next(r for r in doc["results"] if r["id"] == victim)
        self.assertIsNone(broken["score"])
        self.assertIsNone(broken["points_earned"])
        self.assertIn("missing pass(es) [2]", broken["error"])
        intact = next(r for r in doc["results"] if r["id"] != victim)
        self.assertIsNotNone(intact["score"])

    def test_run_skips_aggregation_while_pending(self):
        self.fail_answers.add(self.tasks[0]["id"])
        rc = self._main(*self._run_args("run", "--judge", "judge-a"))
        self.assertEqual(rc, 3)
        self.assertFalse(os.path.exists(self.out))


    # An unparseable judge reply gets exactly one reformat nudge (mirroring
    # the packaged scorer); the recovered verdict scores normally.
    def test_garbled_first_reply_recovers_via_nudge(self):
        self.garble_first_grade = 2
        rc = self._main(*self._run_args("run", "--judge", "judge-a"))
        self.assertEqual(rc, 0)
        self.assertEqual(self.nudges_seen, 2)
        with open(self.out) as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["n_scored"], 2)
        self.assertEqual(doc["summary"]["n_errors"], 0)


    # Checkpoint cells are stamped with a task digest; editing the tasks
    # file must invalidate them rather than apply stale votes to a new rubric.
    def test_edited_rubric_invalidates_recorded_cells(self):
        rc = self._main(*self._run_args("run", "--judge", "judge-a"))
        self.assertEqual(rc, 0)
        edited = os.path.join(self.tmp, "edited-tasks.jsonl")
        with open(FIXTURES) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        tasks[0]["rubricItems"].append(
            {"ordinal": len(tasks[0]["rubricItems"]) + 1,
             "criterionText": "Mentions a brand-new criterion.", "points": 3})
        with open(edited, "w") as f:
            f.writelines(json.dumps(task) + "\n" for task in tasks)
        out2 = os.path.join(self.tmp, "results2.json")
        quiet(hob_eval.main, [
            "aggregate", "--tasks", edited, "--model", "candidate-model",
            "--api-key-env", "HOB_TEST_KEY", "--state-dir", self.state,
            "--out", out2, "--judge", "judge-a"])
        with open(out2) as f:
            doc = json.load(f)
        edited_row = next(r for r in doc["results"] if r["id"] == tasks[0]["id"])
        self.assertIsNone(edited_row["score"])
        self.assertIn("stale", edited_row["error"])
        untouched = next(r for r in doc["results"] if r["id"] == tasks[1]["id"])
        self.assertIsNotNone(untouched["score"])


class TestTaskLoading(unittest.TestCase):
    def test_fixture_ordinals_assigned_in_authored_order(self):
        tasks = hob_eval.load_tasks(FIXTURES)
        self.assertEqual(len(tasks), 2)
        for task in tasks:
            ordinals = [r["ordinal"] for r in task["rubricItems"]]
            self.assertEqual(ordinals, list(range(1, len(ordinals) + 1)))

    def test_missing_field_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write('{"id": "x", "conversation": {"messages": []}}\n')
            path = f.name
        try:
            with self.assertRaises(SystemExit):
                hob_eval.load_tasks(path)
        finally:
            os.unlink(path)

    def test_duplicate_ids_rejected(self):
        line = json.dumps({"id": "x", "conversation": {"messages": []},
                           "rubricItems": [{"criterionText": "c", "points": 1}]})
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(line + "\n" + line + "\n")
            path = f.name
        try:
            with self.assertRaises(SystemExit):
                hob_eval.load_tasks(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
