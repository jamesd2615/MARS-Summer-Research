from __future__ import annotations

import csv
import json
import re
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ResultIntegrityTests(unittest.TestCase):
    def _read_csv(self, relative_path: str) -> list[dict[str, str]]:
        with (ROOT / relative_path).open(newline="", encoding="utf-8-sig") as stream:
            return list(csv.DictReader(stream))

    def test_master_fold_table(self):
        rows = self._read_csv("results/final/final_master_folds.csv")
        self.assertEqual(len(rows), 240)
        keys = {
            (row["dataset"], row["method"], row["experiment_seed"], row["fold"])
            for row in rows
        }
        self.assertEqual(len(keys), len(rows))

    def test_paired_fold_table(self):
        rows = self._read_csv("results/final/final_paired_folds.csv")
        self.assertEqual(len(rows), 200)
        self.assertEqual({row["experiment_seed"] for row in rows}, {"42", "43"})
        keys = {
            (row["dataset"], row["method"], row["experiment_seed"], row["fold"])
            for row in rows
        }
        self.assertEqual(len(keys), len(rows))


class EOHProgramTests(unittest.TestCase):
    def test_all_final_programs_compile(self):
        programs = sorted(
            (ROOT / "programs" / "eoh").glob(
                "*/seed_*/fold_*/selected_program.py"
            )
        )
        self.assertEqual(len(programs), 40)
        for program in programs:
            compile(program.read_text(encoding="utf-8"), str(program), "exec")

    def test_best_sample_provenance_matches_programs(self):
        best_samples = sorted(
            (ROOT / "results" / "search_provenance" / "eoh").glob(
                "*/seed_*/fold_*/best_sample.json"
            )
        )
        self.assertEqual(len(best_samples), 40)

        for sample_path in best_samples:
            relative = sample_path.relative_to(
                ROOT / "results" / "search_provenance" / "eoh"
            )
            dataset, seed, fold = relative.parts[:3]
            program_path = (
                ROOT
                / "programs"
                / "eoh"
                / dataset
                / seed
                / fold
                / "selected_program.py"
            )
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            self.assertEqual(
                sample["code"].strip(),
                program_path.read_text(encoding="utf-8").strip(),
            )


class ArtifactTests(unittest.TestCase):
    def test_figure_manifest_targets_exist(self):
        rows = []
        manifest = ROOT / "results" / "poster_support" / "visualisation_manifest.csv"
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 15)
        for row in rows:
            self.assertTrue((ROOT / row["path"]).is_file(), row["path"])

    def test_presentation_archive(self):
        weekly = sorted((ROOT / "presentations" / "weekly_updates").glob("*.pdf"))
        self.assertEqual(len(weekly), 8)
        for pdf in weekly:
            self.assertEqual(pdf.read_bytes()[:4], b"%PDF")

        poster_pdf = ROOT / "presentations" / "symposium" / "MARS_symposium_poster.pdf"
        self.assertEqual(poster_pdf.read_bytes()[:4], b"%PDF")

        poster_pptx = ROOT / "presentations" / "symposium" / "MARS_symposium_poster.pptx"
        self.assertTrue(zipfile.is_zipfile(poster_pptx))


class PortabilityTests(unittest.TestCase):
    def test_no_personal_absolute_paths(self):
        text_suffixes = {
            ".csv", ".json", ".md", ".py", ".txt", ".ipynb", ".cff"
        }
        forbidden = (
            "C:" + "\\Users\\",
            "OneDrive" + " - Lancaster University",
        )
        failures = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            if path.suffix.lower() not in text_suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(marker in text for marker in forbidden):
                failures.append(str(path.relative_to(ROOT)))
        self.assertEqual(failures, [])

    def test_local_documentation_links_exist(self):
        missing = []
        markdown_files = [ROOT / "README.md"]
        markdown_files.extend((ROOT / "data").glob("*.md"))
        markdown_files.extend((ROOT / "docs").glob("*.md"))
        markdown_files.extend((ROOT / "figures").glob("*.md"))
        markdown_files.extend((ROOT / "notebooks").glob("*.md"))
        markdown_files.extend((ROOT / "presentations").glob("*.md"))
        markdown_files.extend((ROOT / "programs").glob("**/*.md"))
        markdown_files.extend((ROOT / "results").glob("*.md"))

        for document in markdown_files:
            text = document.read_text(encoding="utf-8")
            local_links = re.findall(r"\[[^]]+\]\((?!https?://)([^)#]+)", text)
            for link in local_links:
                if not (document.parent / link).resolve().exists():
                    missing.append(f"{document.relative_to(ROOT)} -> {link}")
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
