import pandas as pd
import re
import unicodedata
from pathlib import Path


INPUT_FILE = Path("data/processed/job_skills_normalized.csv")
OUTPUT_FILE = Path("data/processed/job_skills_canonical.csv")


# ---------------------------------------------------------
# High-confidence canonical mappings
# ---------------------------------------------------------
# Only merge variants that clearly represent the same skill.
# Do NOT merge parent skills with specific technologies.
# ---------------------------------------------------------

CANONICAL_ALIASES = {

    # Microsoft Office
    "microsoft-office": "microsoft office",
    "microsoftoffice": "microsoft office",

    # Frontend
    "front end": "frontend",
    "front-end": "frontend",

    # Backend
    "back-end": "backend",

    # Full stack
    "full stack": "full-stack",
    "fullstack": "full-stack",

    "full stack development": "full-stack development",
    "fullstack development": "full-stack development",

    # Project Management
    "project-management": "project management",
    "projectmanagement": "project management",

    # Customer Service
    "customer-service": "customer service",
    "customerservice": "customer service",

    # Customer Engagement
    "customer-engagement": "customer engagement",
    "customerengagement": "customer engagement",

    # Customer Relations
    "customer-relations": "customer relations",
    "customerrelations": "customer relations",

    # Cold Calling
    "cold-calling": "cold calling",
    "coldcalling": "cold calling",

    # Time Management
    "time-management": "time management",
    "timemanagement": "time management",

    # Stakeholder Management
    "stakeholder-management": "stakeholder management",
    "stakeholdermanagement": "stakeholder management",

    # Strategic Planning
    "strategic-planning": "strategic planning",
    "strategicplanning": "strategic planning",

    # Relationship Building
    "relationship-building": "relationship building",
    "relationshipbuilding": "relationship building",

    # Record Keeping
    "record-keeping": "record keeping",
    "recordkeeping": "record keeping",

    # Quality Improvement
    "quality-improvement": "quality improvement",
    "qualityimprovement": "quality improvement",

    # Data Driven
    "data-driven": "data driven",
    "datadriven": "data driven",

    # Patient Care
    "patient-care": "patient care",
    "patientcare": "patient care",

    # Access Control
    "accesscontrol": "access control",

    # Account Management
    "account-management": "account management",

    # Active Directory
    "activedirectory": "active directory",

    # API Management
    "apimanagement": "api management",

    # ArcGIS
    "arc-gis": "arcgis",

    # Argo
    "argo-cd": "argo cd",
    "argocd": "argo cd",

    "argo-workflows": "argo workflows",
    "argoworkflows": "argo workflows",

    # AWS GovCloud
    "aws gov cloud": "aws govcloud",

    # Basic Life Support
    "basic-life-support": "basic life support",

    # Bar Code
    "bar-code-scanning": "barcode scanning",

    # Lean Six Sigma
    "lean sixsigma": "lean six sigma",
    "lean/six sigma": "lean six sigma",
    "leansixsigma": "lean six sigma",

    # Ab Initio
    "ab-initio": "ab initio",
    "abinitio": "ab initio",

    # Ad Tech
    "ad-tech": "ad tech",
    "adtech": "ad tech",

    # QA/QC
    "qa qc": "qa/qc",
    "qaqc": "qa/qc",

    # AI tools
    "ai-tools": "ai tools",
    "aitools": "ai tools",

    # Cost Benefit Analysis
    "cost-benefit analysis": "cost benefit analysis",
    "cost/benefit analysis": "cost benefit analysis",

    # Financial Record Keeping
    "financial record-keeping": "financial record keeping",
    "financial recordkeeping": "financial record keeping",

    # Fire Alarm
    "fire-alarm": "fire alarm",
    "firealarm": "fire alarm",

    # HVAC
    "hvac/r": "hvac-r",
    "hvacr": "hvac-r",

    # ML Ops
    "ml-ops": "mlops",

    # Omni Channel
    "omni-channel": "omnichannel",

    # Pro/E
    "pro/e": "pro-e",
    "proe": "pro-e",

    # Security Compliance
    "security/compliance": "security compliance",
    "securitycompliance": "security compliance",

    # Team Lead
    "team-lead": "team lead",
    "teamlead": "team lead",

    # US GAAP
    "us gaap": "u.s. gaap",
    "usgaap": "u.s. gaap",

    # 3D CAD
    "3d-cad": "3d cad",

    # 6 Sigma
    "6sigma": "6 sigma",

    # Active Directory / similar formatting
    "adapterconfig": "adapter config",

    # Adobe CC
    "adobecc": "adobe cc",

    # Agile / DevOps
    # These remain distinct combinations and are NOT merged.
}


def normalize_text(value):
    """Normalize Unicode, whitespace and case."""

    if not isinstance(value, str):
        return ""

    value = unicodedata.normalize("NFKC", value)

    value = value.replace("\u2011", "-")
    value = value.replace("\u2013", "-")
    value = value.replace("\u2014", "-")

    value = re.sub(r"\s+", " ", value)

    return value.strip().lower()


def canonicalize(skill):
    """Return the controlled canonical skill name."""

    normalized = normalize_text(skill)

    if not normalized:
        return ""

    return CANONICAL_ALIASES.get(
        normalized,
        normalized
    )


print("Loading normalized skills...")

df = pd.read_csv(INPUT_FILE)

print(f"Skill mappings loaded: {len(df)}")


# Apply controlled consolidation
df["canonical_skill"] = df["canonical_skill"].apply(
    canonicalize
)


# Remove duplicate mappings
df = df.drop_duplicates(
    subset=[
        "original_skill",
        "normalized_skill",
        "canonical_skill"
    ]
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 70)
print("CONTROLLED CONSOLIDATION COMPLETE")
print("=" * 70)

print(
    f"Unique skills before consolidation: "
    f"{df['canonical_skill'].nunique()}"
)


# Display mappings that were actually changed
changed = df[
    df["normalized_skill"] != df["canonical_skill"]
]


print(
    f"Mappings changed: {len(changed)}"
)

print(
    f"Final canonical skills: "
    f"{df['canonical_skill'].nunique()}"
)


print("\nSample consolidated mappings:")

sample = changed[
    [
        "normalized_skill",
        "canonical_skill"
    ]
].drop_duplicates().head(40)


if len(sample) > 0:
    print(
        sample.to_string(index=False)
    )
else:
    print("No additional mappings were changed.")


print(
    f"\nSaved to: {OUTPUT_FILE}"
)