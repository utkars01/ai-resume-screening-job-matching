from pathlib import Path
import re
import unicodedata

import pandas as pd


MASTER_FILE = Path("data/processed/skill_master.csv")


class SkillService:
    def __init__(self):
        if not MASTER_FILE.exists():
            raise FileNotFoundError(
                f"Skill master file not found: {MASTER_FILE}"
            )

        master = pd.read_csv(MASTER_FILE)

        required_columns = {
            "normalized_skill",
            "canonical_skill",
        }

        missing_columns = required_columns - set(master.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in skill master: {missing_columns}"
            )

        self.skill_mapping = (
            master[
                ["normalized_skill", "canonical_skill"]
            ]
            .dropna()
            .drop_duplicates("normalized_skill")
            .set_index("normalized_skill")[
                "canonical_skill"
            ]
            .to_dict()
        )

        # Match longer and more specific skills first.
        self.skills = sorted(
            self.skill_mapping.keys(),
            key=len,
            reverse=True,
        )
        # Compile once instead of rebuilding every regex on every extraction.
        self.skill_patterns = [
            (
                skill,
                re.compile(rf"(?<!\w){re.escape(skill)}(?!\w)", re.IGNORECASE),
            )
            for skill in self.skills
        ]

    @staticmethod
    def normalize_skill(skill: str) -> str:
        """
        Normalize text using the same basic normalization
        strategy used by the existing dataset pipeline.
        """

        skill = unicodedata.normalize(
            "NFKC",
            skill,
        )

        skill = skill.replace("\u2011", "-")
        skill = skill.replace("\u2013", "-")
        skill = skill.replace("\u2014", "-")

        skill = skill.strip().lower()

        skill = re.sub(
            r"\s+",
            " ",
            skill,
        )

        return skill

    def extract_skills(
        self,
        text: str,
    ) -> list[str]:
        """
        Extract canonical skills from resume text.

        Longer/more specific skills are preferred over
        shorter skills that occur inside them.
        """

        if not text or not text.strip():
            return []

        normalized_text = self.normalize_skill(text)

        matched_skills = []

        # -----------------------------------------------------
        # Find all skills in the resume text
        # -----------------------------------------------------
        for skill, pattern in self.skill_patterns:
            if pattern.search(normalized_text):
                canonical = self.skill_mapping[skill]

                matched_skills.append(
                    (
                        skill,
                        canonical,
                    )
                )

        # -----------------------------------------------------
        # Remove generic sub-skills when a more specific
        # skill contains them.
        #
        # Example:
        # "machine learning"
        # should be preferred over "learning".
        # -----------------------------------------------------
        selected = []

        for skill, canonical in matched_skills:

            is_subskill = False

            for selected_skill, _ in selected:

                if (
                    skill != selected_skill
                    and skill in selected_skill
                ):
                    is_subskill = True
                    break

            if not is_subskill:
                selected.append(
                    (
                        skill,
                        canonical,
                    )
                )

        # -----------------------------------------------------
        # Return unique canonical skills
        # -----------------------------------------------------
        return sorted(
            {
                canonical
                for _, canonical in selected
            }
        )


# ---------------------------------------------------------
# Shared service instance
# ---------------------------------------------------------

skill_service = SkillService()
