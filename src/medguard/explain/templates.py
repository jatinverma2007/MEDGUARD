"""Bilingual explanation templates.

Every sentence produced here is a *template filled with retrieved evidence*.
There is no free-form generation, which means:

* every claim is traceable to a knowledge-base row,
* no hallucination is possible,
* the output is deterministic and testable,
* the Hindi version is not a translation produced at runtime.

The optional LLM layer in :mod:`medguard.explain.llm` can enrich the text,
but it must quote the same evidence and is verified with an NLI entailment
check before it is shown.
"""

from __future__ import annotations

from medguard.schemas import Explanation, InteractionFinding, Severity

SEVERITY_LABEL_EN: dict[Severity, str] = {
    Severity.CONTRAINDICATED: "Contraindicated",
    Severity.MAJOR: "Major",
    Severity.MODERATE: "Moderate",
    Severity.MINOR: "Minor",
    Severity.NONE: "No known interaction",
    Severity.UNKNOWN: "Unknown",
}

SEVERITY_LABEL_HI: dict[Severity, str] = {
    Severity.CONTRAINDICATED: "वर्जित (Contraindicated)",
    Severity.MAJOR: "गंभीर (Major)",
    Severity.MODERATE: "मध्यम (Moderate)",
    Severity.MINOR: "हल्का (Minor)",
    Severity.NONE: "कोई ज्ञात अंतरिक्रिया नहीं",
    Severity.UNKNOWN: "अज्ञात",
}

#: Plain-language mechanism descriptions keyed by event type.
MECHANISM_EN: dict[str, str] = {
    "bleeding": "both medicines slow blood clotting, so bleeding risk rises",
    "gi bleeding": "both medicines irritate the stomach lining and slow blood clotting",
    "hypoglycemia": "together they can push blood sugar too low",
    "hyperglycemia": "together they can reduce the medicines' effectiveness",
    "sedation": "both slow down the central nervous system",
    "drowsiness": "both slow down the central nervous system",
    "dizziness": "both can lower blood pressure or affect alertness",
    "hypotension": "together they lower blood pressure more than expected",
    "hypertension": "together they raise blood pressure more than expected",
    "renal impairment": "together they put extra load on the kidneys",
    "hepatotoxicity": "together they stress the liver",
    "neurotoxicity": "together they affect the nervous system",
    "seizure": "together they lower the seizure threshold",
    "qt prolongation": "together they affect heart rhythm",
    "torsades de pointes": "together they can cause a dangerous heart rhythm",
    "serotonin syndrome": "together they increase serotonin activity",
    "hyperkalemia": "together they raise potassium levels",
    "respiratory depression": "together they slow breathing",
    "cushing syndrome": "together they increase steroid exposure",
}

MECHANISM_HI: dict[str, str] = {
    "bleeding": "दोनों दवाइयाँ रक्त को जमने में मदद करती हैं, इसलिए रक्तस्राव का खतरा बढ़ता है",
    "gi bleeding": "दोनों दवाइयाँ पेट की झिल्ली को नुकसान पहुँचाती हैं और रक्तस्राव का खतरा बढ़ाती हैं",
    "hypoglycemia": "दोनों साथ में शुगर को बहुत नीचे ले जा सकती हैं",
    "hyperglycemia": "दोनों साथ में दवा का असर कम कर सकती हैं",
    "sedation": "दोनों केंद्रीय तंत्रिका तंत्र को धीमा करती हैं",
    "drowsiness": "दोनों केंद्रीय तंत्रिका तंत्र को धीमा करती हैं",
    "dizziness": "दोनों रक्तचाप गिरा सकती हैं या चेतना को प्रभावित कर सकती हैं",
    "hypotension": "दोनों मिलकर रक्तचाप को अपेक्षा से अधिक गिरा सकती हैं",
    "hypertension": "दोनों मिलकर रक्तचाप को अपेक्षा से अधिक बढ़ा सकती हैं",
    "renal impairment": "दोनों मिलकर गुर्दों पर अतिरिक्त भार डालती हैं",
    "hepatotoxicity": "दोनों मिलकर यकृत (लीवर) पर तनाव डालती हैं",
    "neurotoxicity": "दोनों मिलकर तंत्रिका तंत्र को प्रभावित करती हैं",
    "seizure": "दोनों मिलकर दौरे की सीमा कम कर सकती हैं",
    "qt prolongation": "दोनों मिलकर हृदय की धड़कन को प्रभावित करती हैं",
    "torsades de pointes": "दोनों मिलकर खतरनाक हृदय लय उत्पन्न कर सकती हैं",
    "serotonin syndrome": "दोनों मिलकर सेरोटोनिन की सक्रियता बढ़ाती हैं",
    "hyperkalemia": "दोनों मिलकर पोटैशियम का स्तर बढ़ा सकती हैं",
    "respiratory depression": "दोनों मिलकर साँस धीमी कर सकती हैं",
    "cushing syndrome": "दोनों मिलकर स्टेरॉयड की मात्रा बढ़ा सकती हैं",
}

#: Closing advice shown for each severity, in both languages.
ADVICE_EN: dict[Severity, str] = {
    Severity.CONTRAINDICATED: (
        "Do not take these two medicines together without your prescriber's explicit "
        "instruction. Ask them for an alternative."
    ),
    Severity.MAJOR: (
        "Contact your doctor or pharmacist before your next dose. They may need to "
        "change the dose or choose a different medicine, and may want to monitor you."
    ),
    Severity.MODERATE: (
        "Tell your doctor or pharmacist about this combination. Extra monitoring or a "
        "dose adjustment may be needed."
    ),
    Severity.MINOR: (
        "Mention this combination to your pharmacist, and watch for the listed side "
        "effects. It is usually manageable."
    ),
    Severity.NONE: "No known interaction between these two medicines.",
    Severity.UNKNOWN: (
        "MedGuard does not have enough evidence about this combination. Please check "
        "with your doctor or pharmacist."
    ),
}

ADVICE_HI: dict[Severity, str] = {
    Severity.CONTRAINDICATED: (
        "अपने डॉक्टर की स्पष्ट सलाह के बिना ये दोनों दवाइयाँ साथ न लें। वैकल्पिक दवा पूछें।"
    ),
    Severity.MAJOR: (
        "अगली खुराक से पहले अपने डॉक्टर या फार्मेसिस्ट से संपर्क करें। उन्हें खुराक बदलनी या "
        "कोई दूसरी दवा चुननी पड़ सकती है, और निगरानी आवश्यक हो सकती है।"
    ),
    Severity.MODERATE: (
        "इस संयोजन के बारे में अपने डॉक्टर या फार्मेसिस्ट को बताएँ। अतिरिक्त निगरानी या खुराक "
        "समायोजन आवश्यक हो सकता है।"
    ),
    Severity.MINOR: (
        "यह संयोजन अपने फार्मेसिस्ट को बताएँ और बताए गए दुष्प्रभावों पर ध्यान दें। आमतौर पर यह नियंत्रण में रहता है।"
    ),
    Severity.NONE: "इन दो दवाइयों के बीच कोई ज्ञात अंतरिक्रिया नहीं है।",
    Severity.UNKNOWN: (
        "MedGuard के पास इस संयोजन के बारे में पर्याप्त जानकारी नहीं है। कृपया अपने डॉक्टर या फार्मेसिस्ट से जाँच करें।"
    ),
}

DISCLAIMER_EN = (
    "This is decision-support information, not medical advice. It has not been reviewed "
    "by a clinician. Always confirm with your doctor or pharmacist before changing how "
    "you take any medicine."
)

DISCLAIMER_HI = (
    "यह केवल सहायता जानकारी है, चिकित्सा सलाह नहीं। इसकी किसी चिकित्सक द्वारा समीक्षा नहीं की गई "
    "है। दवा लेने का तरीका बदलने से पहले हमेशा अपने डॉक्टर या फार्मेसिस्ट से पुष्टि करें।"
)


def _mechanism(event_type: str | None, language: str) -> str:
    if not event_type:
        return ""
    table = MECHANISM_EN if language == "en" else MECHANISM_HI
    lowered = event_type.strip().lower()
    for key, text in table.items():
        if key in lowered:
            return text
    return event_type


def explain_finding(finding: InteractionFinding, language: str = "en") -> Explanation:
    """Build a grounded, templated explanation for one finding."""
    language = "hi" if language == "hi" else "en"

    labels = SEVERITY_LABEL_EN if language == "en" else SEVERITY_LABEL_HI
    advice = ADVICE_EN if language == "en" else ADVICE_HI
    disclaimer = DISCLAIMER_EN if language == "en" else DISCLAIMER_HI

    drug_a = finding.drug_a
    drug_b = finding.drug_b

    if language == "en":
        parts = [
            f"**{drug_a.capitalize()}** and **{drug_b.capitalize()}** - "
            f"{labels[finding.severity]} interaction."
        ]
        mechanism = _mechanism(finding.event_type, language)
        if mechanism:
            parts.append(f"Mechanism: {mechanism}.")
        if finding.description:
            parts.append(finding.description.rstrip(".") + ".")
        parts.append(advice[finding.severity])
        parts.append(disclaimer)
        text = "\n\n".join(parts)
    else:
        parts = [f"**{drug_a}** और **{drug_b}** - {labels[finding.severity]} अंतरिक्रिया।"]
        mechanism = _mechanism(finding.event_type, language)
        if mechanism:
            parts.append(f"कारण: {mechanism}।")
        if finding.description:
            parts.append(finding.description.rstrip(".") + "।")
        parts.append(advice[finding.severity])
        parts.append(disclaimer)
        text = "\n\n".join(parts)

    return Explanation(
        text=text,
        language=language,
        grounded=True,
        citations=list(finding.evidence),
    )


def explain_all(findings: list[InteractionFinding], language: str = "en") -> list[Explanation]:
    """Explain every finding."""
    return [explain_finding(f, language=language) for f in findings]


def overall_summary(
    findings: list[InteractionFinding],
    n_drugs: int,
    language: str = "en",
) -> Explanation:
    """A one-paragraph summary of the whole prescription."""
    language = "hi" if language == "hi" else "en"

    if not findings:
        text = (
            f"Checked {n_drugs} medicines. No known interactions were found between them.\n\n"
            f"{DISCLAIMER_EN}"
            if language == "en"
            else f"{n_drugs} दवाइयों की जाँच की गई। इनके बीच कोई ज्ञात अंतरिक्रिया नहीं मिली।\n\n{DISCLAIMER_HI}"
        )
        return Explanation(text=text, language=language, grounded=True)

    counts: dict[Severity, int] = {}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1

    labels = SEVERITY_LABEL_EN if language == "en" else SEVERITY_LABEL_HI
    breakdown = ", ".join(
        f"{counts[sev]}x {labels[sev]}"
        for sev in sorted(counts, key=lambda s: -s.rank)
        if counts[sev]
    )

    if language == "en":
        text = (
            f"Checked {n_drugs} medicines and found {len(findings)} interaction(s): {breakdown}. "
            f"Review each finding below and speak to your doctor or pharmacist before making "
            f"any change.\n\n{DISCLAIMER_EN}"
        )
    else:
        text = (
            f"{n_drugs} दवाइयों की जाँच की गई और {len(findings)} अंतरिक्रिया मिलीं: {breakdown}। "
            f"किसी भी बदलाव से पहले नीचे दी गई हर जानकारी देखें और अपने डॉक्टर या फार्मेसिस्ट से "
            f"बात करें।\n\n{DISCLAIMER_HI}"
        )

    return Explanation(text=text, language=language, grounded=True)
