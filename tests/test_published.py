"""Published statements: parsing, thresholds, the earlier-release prior, one question and the gate of the run."""
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import published as p  # noqa: E402
from connectomequest.annotation_v35 import engine as e  # noqa: E402

# Results of the single run, the generated numbers and the list of papers searched live outside version control.
needs_run = pytest.mark.skipif(
    not all(f.exists() for f in (p.OUT / "questions.npz", p.OUT / "results.json", ROOT / "paper/published_numbers.tex")),
    reason="needs the outputs of the run and the generated numbers")
needs_names = pytest.mark.skipif(not (p.OUT / "names.json").exists(), reason="needs outputs/published/names.json")
needs_searched = pytest.mark.skipif(not (ROOT / "inputs/published/searched.json").exists(),
                                    reason="needs inputs/published/searched.json")

VOCABULARY = ["<outside>", "KCab-c", "KCab-m", "KCg-m", "MBON01", "MBON02", "EPG"]


def statement(**changes):
    base = dict(id="s1", source="li2020mushroom", doi="10.7554/eLife.62576", where="Results", quote="A sentence.", label="KC to MBON",
                partner=["KC*"], target=["MBON01"], percent="40")
    base.update(changes)
    return base


def write(tmp_path, statements):
    path = tmp_path / "statements.json"
    path.write_text(json.dumps(dict(statements=statements)))
    return path


def test_statements_are_validated(tmp_path):
    assert [s["id"] for s in p.read_statements(write(tmp_path, [statement(), statement(id="s2")]))] == ["s1", "s2"]
    for bad in (statement(percent="0"), statement(percent="120"), statement(quote=" "), statement(partner=[])):
        with pytest.raises(ValueError):
            p.read_statements(write(tmp_path, [bad]))
    with pytest.raises(ValueError):
        p.read_statements(write(tmp_path, [statement(), statement()]))
    missing = statement()
    del missing["doi"]
    with pytest.raises(ValueError):
        p.read_statements(write(tmp_path, [missing]))


def test_names_and_prefixes_select_types():
    assert p.codes_of(["KC*"], VOCABULARY).tolist() == [1, 2, 3]
    assert p.codes_of(["KCab*", "EPG"], VOCABULARY).tolist() == [1, 2, 6]
    assert p.codes_of(["MBON01"], VOCABULARY).tolist() == [4]
    with pytest.raises(ValueError):
        p.codes_of(["MBON9"], VOCABULARY)
    with pytest.raises(ValueError):
        p.codes_of(["<outside>"], VOCABULARY)  # code 0 is never a type


@pytest.mark.parametrize("total", [1, 7, 10, 999, 12345])
@pytest.mark.parametrize("percent", ["40", "12.5", "85", "100", 33])
def test_threshold_is_the_exact_ceiling(total, percent):
    for numerator, denominator in ((1, 1), (1, 2)):
        exact = Fraction(str(percent)) / 100 * Fraction(numerator, denominator) * total
        expected = -(-exact.numerator // exact.denominator)
        assert p.threshold(total, percent, numerator, denominator) == expected


def test_earlier_prior_puts_its_weight_on_the_earlier_label():
    topology = np.full((4, 7), 1 / 7, dtype=np.float32)
    hint = np.array([1, 5, 0, -1])  # a partner type, another type, a name outside the vocabulary, no earlier label
    prior = p.earlier_prior(topology, hint, np.array([1, 2, 3]), 0.9)
    np.testing.assert_allclose(prior.sum(1), 1.0, atol=1e-6)
    assert prior[0, 0] == pytest.approx(0.9 + 0.1 * 3 / 7, abs=1e-6)  # member
    assert prior[1, 1] == pytest.approx(0.9 + 0.1 * 3 / 7, abs=1e-6)  # nonmember
    assert prior[2, 2] == pytest.approx(0.9 + 0.1 / 7, abs=1e-6)  # unresolved
    np.testing.assert_allclose(prior[3], [3 / 7, 3 / 7, 1 / 7], atol=1e-6)  # topology alone


def question():
    cells = np.arange(1, 7)
    weights = np.array([50, 20, 10, 10, 5, 5])
    outcomes = np.array([e.MEMBER, e.NONMEMBER, e.MEMBER, e.UNRESOLVED, e.NONMEMBER, e.MEMBER])
    prior = np.tile((0.5, 0.5, 0.0), (6, 1))
    return cells, weights, outcomes, prior


def test_one_question_reports_answer_bound_and_costs():
    cells, weights, outcomes, prior = question()
    truth, bound, costs = p.ask(cells, weights, outcomes, prior, 50)  # the heaviest member certifies yes
    assert (truth, bound, costs["weight"]) == ("positive", 1, 1)
    truth, bound, costs = p.ask(cells, weights, outcomes, prior, 66)  # members carry 65, the unresolved cell 10
    assert truth == "unresolved" and bound is None and set(costs.values()) == {6}
    truth, bound, costs = p.ask(cells, weights, outcomes, prior, 80)  # U = 75 < 80 once both nonmembers are checked
    assert truth == "negative" and bound == 2 and all(cost >= bound for cost in costs.values())


def test_summary_counts_answers_and_adds_checks_of_certifiable_questions():
    rows = [dict(truth=1, size=6, clairvoyant=1, weight=1, direction=1, ag=2, recal=1),
            dict(truth=2, size=8, clairvoyant=2, weight=4, direction=3, ag=4, recal=3),
            dict(truth=0, size=9, clairvoyant=-1, weight=9, direction=9, ag=9, recal=9)]
    s = p.summarize(rows)
    assert (s["neurons"], s["yes"], s["no"], s["uncertifiable"]) == (3, 1, 1, 1)
    assert (s["weight"], s["direction"], s["ag"], s["recal"], s["clairvoyant"], s["input_cells_certifiable"]) == (5, 4, 6, 4, 3, 14)
    assert p.summarize([rows[2]]).get("weight") is None


def test_the_run_needs_the_record_and_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUT", tmp_path)
    monkeypatch.setattr(p, "RECORD", tmp_path / "record.json")
    with pytest.raises(SystemExit, match="needs the record"):
        p.run(None)
    (tmp_path / "record.json").write_text(json.dumps(dict(code={}, data={})))
    (tmp_path / "results.json").write_text("{}")
    with pytest.raises(SystemExit, match="result exists"):
        p.run(None)
    (tmp_path / "results.json").unlink()
    (tmp_path / "claim").write_text("taken\n")
    with pytest.raises(FileExistsError):
        p.run(None)


def test_a_changed_file_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "OUT", tmp_path)
    monkeypatch.setattr(p, "RECORD", tmp_path / "record.json")
    (tmp_path / "record.json").write_text(json.dumps(dict(code={"scripts/published.py": "0" * 64}, data={})))
    with pytest.raises(SystemExit, match="differs from the record"):
        p.run(None)


def test_source_shares_count_all_pairs_and_the_pairs_kept_by_the_threshold():
    import published_source as source

    pre = np.array([10, 11, 12, 13, 20, 10, 20])
    post = np.array([20, 20, 20, 20, 20, 21, 10])
    weight = np.array([8, 3, 5, 4, 9, 6, 7])  # the self-connection of 20 and the edges out of it are not its input
    labels_of = {10: 1, 11: 1, 12: 2, 13: 3, 20: 4, 21: 5}
    r = source.shares(pre, post, weight, labels_of, partner={1}, target={4}, minimum=5)
    assert r == dict(neurons=1, synapses=20, partner_synapses=11, synapses_kept=13, partner_synapses_kept=8)


@needs_names
def test_names_count_the_types_that_share_a_stem_and_what_the_release_declares():
    import published_names as names

    types = ["pC1x_a", "pC1x_a", "pC1_2", None, "MBON01", "aIPg1"]
    declared = ["pC1a,pC1c", None, "pC1b", "pC1d,pC1e", None, "aIPg1"]
    row = names.naming(types, declared, "pC1d")
    assert (row["neurons"], row["types_sharing_stem"], row["neurons_sharing_stem"]) == (0, 2, 3)
    assert row["declared_hemibrain_types"] == {"pC1a,pC1c": 1, "pC1b": 1} and row["declared_as_type"] == 1
    stored = {r["type"]: r for r in json.loads((p.OUT / "names.json").read_text())["types"]}
    assert set(stored) == {"pC1d", "aIPg3"} and all(r["neurons"] == 0 and r["declared_as_type"] == 0 for r in stored.values())


@needs_run
def test_the_no_answers_hold_for_any_reading_of_nearly_half():
    q = np.load(p.OUT / "questions.npz")
    result = json.loads((p.OUT / "results.json").read_text())
    order = [s["id"] for s in result["statements"]]
    rows = (q["level"] == 0) & (q["statement"] == order.index("kcgs2-kc")) & (q["view"] == result["views"].index("flywire"))
    assert rows.sum() == 2 and (q["truth"][rows] == p.STATUS["negative"]).all()
    assert ((q["member"] + q["unresolved"])[rows] / q["total"][rows]).max() < 0.35
    macros = (ROOT / "paper/published_numbers.tex").read_text()
    assert "\\newcommand{\\PubKcFlyUpperPct}{33}" in macros and "\\newcommand{\\PubPolicies}{four}" in macros


@needs_run
def test_every_open_question_lacks_partner_input_that_unresolved_cells_could_supply():
    q = np.load(p.OUT / "questions.npz")
    result = json.loads((p.OUT / "results.json").read_text())
    rows = (q["level"] == 0) & (q["truth"] == p.STATUS["unresolved"])
    assert rows.sum() == 9 and (rows & (q["view"] == result["views"].index("malecns"))).sum() == 7
    assert (q["member"][rows] < q["threshold"][rows]).all() and ((q["member"] + q["unresolved"])[rows] >= q["threshold"][rows]).all()
    macros = (ROOT / "paper/published_numbers.tex").read_text()
    assert "\\newcommand{\\PubOpenMale}{seven}" in macros and "\\newcommand{\\PubSearched}{twenty}" in macros


@needs_searched
def test_the_papers_searched_are_listed_with_the_sources_of_the_statements():
    searched = json.loads((ROOT / "inputs/published/searched.json").read_text())["papers"]
    assert len(searched) == 20 and len({s["pmcid"] for s in searched}) == 20
    assert all(s["pmcid"].startswith("PMC") and s["doi"] and s["title"] and s["journal"] for s in searched)
    assert {s["doi"] for s in p.read_statements(ROOT / p.STATEMENTS)} <= {s["doi"] for s in searched}


def test_tables_have_one_row_per_statement_and_format_shares():
    import published_assets as a

    assert [a.share(x) for x in (None, 0.0, 0.007, 0.0994, 0.0996, 0.413)] == ["---", "0", "0.7", "9.9", "10", "41"]
    level = dict(neurons=2, yes=1, no=0, uncertifiable=1, weight=7, direction=4)  # checks at the published share
    empty = dict(neurons=0, lower=None, upper=None, levels={})
    asked = dict(neurons=2, lower=0.15, upper=0.47, levels=dict(published=level, half=dict(level, yes=2, uncertifiable=0)))
    result = dict(views=["malecns", "flywire"], statements=[dict(id="s1", hemibrain=dict(share=0.19), views=dict(malecns=asked, flywire=empty))])
    main, full = a.tables(dict(s1=statement(percent="15")), result)
    assert main == [r"KC to MBON & 15 & 19 & 15--47 & 1/0/1 & --- & ---\\"]
    assert full == [r"KC to MBON & 2 & 2/0/0 & 7/4 & 0 & --- & ---\\"]
