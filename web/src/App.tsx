import { useState, useCallback } from 'react'

type Language = 'en' | 'hi'

interface Drug {
  surface: string
  normalized: string | null
  canonical_name: string | null
  dose: string | null
  confidence: number | null
}

interface Finding {
  drug_a: string
  drug_b: string
  severity: 'none' | 'minor' | 'moderate' | 'major' | 'contraindicated' | 'unknown'
  event_type: string | null
  description: string | null
  management: string | null
  evidence: string[]
  source: 'kb' | 'model' | 'both'
  confidence: { raw: number; calibrated: number; temperature: number } | null
}

interface Explanation {
  text: string
  language: 'en' | 'hi'
  grounded: boolean
  citations: string[]
}

interface AnalysisResult {
  drugs: Drug[]
  findings: Finding[]
  explanations: Explanation[]
  overall_severity: 'none' | 'minor' | 'moderate' | 'major' | 'contraindicated' | 'unknown'
  abstention: {
    reason: string
    threshold: number | null
    confidence: number | null
    details: Record<string, string | number>
  } | null
  processing_ms: number | null
}

const API_BASE = '/api'

const SEVERITY_COLORS: Record<string, string> = {
  none: '#16a34a',
  minor: '#22c55e',
  moderate: '#d97706',
  major: '#dc2626',
  contraindicated: '#991b1b',
  unknown: '#64748b',
}

const SEVERITY_LABELS: Record<string, { en: string; hi: string }> = {
  none: { en: 'No interaction', hi: 'कोई अंतरिक्रिया नहीं' },
  minor: { en: 'Minor', hi: 'हल्का' },
  moderate: { en: 'Moderate', hi: 'मध्यम' },
  major: { en: 'Major', hi: 'गंभीर' },
  contraindicated: { en: 'Contraindicated', hi: 'वर्जित' },
  unknown: { en: 'Unknown', hi: 'अज्ञात' },
}

function SeverityBadge({ severity, language }: { severity: string; language: Language }) {
  const color = SEVERITY_COLORS[severity] || '#64748b'
  const label = SEVERITY_LABELS[severity]?.[language] || severity
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '4px 10px',
        borderRadius: '9999px',
        fontSize: '0.75rem',
        fontWeight: 600,
        background: `${color}15`,
        color,
        border: `1px solid ${color}40`,
      }}
    >
      {label}
    </span>
  )
}

function DrugRow({ drug }: { drug: Drug }) {
  return (
    <tr style={{ borderBottom: '1px solid var(--border)' }}>
      <td style={{ padding: '12px 16px' }}>
        <strong>{drug.normalized || drug.surface}</strong>
        {drug.surface !== (drug.normalized || drug.surface) && (
          <span style={{ marginLeft: 8, fontSize: '0.8rem', color: 'var(--muted)' }}>
            ({drug.surface})
          </span>
        )}
      </td>
      <td style={{ padding: '12px 16px', color: 'var(--muted)' }}>{drug.dose || '—'}</td>
      <td style={{ padding: '12px 16px' }}>
        {drug.confidence !== null && (
          <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>
            {(drug.confidence * 100).toFixed(0)}%
          </span>
        )}
      </td>
    </tr>
  )
}

function FindingCard({ finding, language, index }: { finding: Finding; language: Language; index: number }) {
  return (
    <div
      key={index}
      style={{
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '16px',
        marginBottom: '12px',
        background: 'var(--card)',
        boxShadow: 'var(--shadow)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
        <code style={{
          fontSize: '0.9rem', padding: '4px 8px', borderRadius: '6px',
          background: 'var(--bg)', border: '1px solid var(--border)',
          fontFamily: 'monospace',
        }}>
          {finding.drug_a} + {finding.drug_b}
        </code>
        <SeverityBadge severity={finding.severity} language={language} />
        <span style={{ marginLeft: 'auto', fontSize: '0.75rem', color: 'var(--muted)' }}>
          {finding.source === 'kb' ? 'Knowledge base' : finding.source === 'model' ? 'Model prediction' : 'Both'}
        </span>
      </div>

      {finding.event_type && (
        <div style={{ marginBottom: 8 }}>
          <span style={{ fontWeight: 500, color: 'var(--muted)', fontSize: '0.85rem' }}>
            {language === 'en' ? 'Mechanism:' : 'कारण:'}
          </span>
          <span style={{ marginLeft: 8 }}>{finding.event_type}</span>
        </div>
      )}

      {finding.description && (
        <div style={{ marginBottom: 8 }}>
          <span style={{ fontWeight: 500, color: 'var(--muted)', fontSize: '0.85rem' }}>
            {language === 'en' ? 'Description:' : 'विवरण:'}
          </span>
          <span style={{ marginLeft: 8 }}>{finding.description}</span>
        </div>
      )}

      {finding.confidence && (
        <div style={{ marginBottom: 8, fontSize: '0.8rem', color: 'var(--muted)' }}>
          {language === 'en' ? 'Calibrated confidence:' : 'कैलिब्रेटेड विश्वास:'}
          {' '}
          {(finding.confidence.calibrated * 100).toFixed(1)}%
          {finding.confidence.temperature !== 1.0 && (
            <> (T = {finding.confidence.temperature.toFixed(3)})</>
          )}
        </div>
      )}

      {finding.evidence.length > 0 && (
        <details style={{ marginTop: 8 }}>
          <summary style={{ fontSize: '0.8rem', color: 'var(--muted)', cursor: 'pointer' }}>
            {language === 'en' ? 'Evidence sources' : 'स्रोत'}
          </summary>
          <ul style={{ marginTop: 8, paddingLeft: 20, fontSize: '0.8rem', color: 'var(--muted)' }}>
            {finding.evidence.map((e, i) => <li key={i}>{e}</li>)}
          </ul>
        </details>
      )}
    </div>
  )
}

function ExplanationCard({ explanation, language }: { explanation: Explanation; language: Language }) {
  return (
    <div
      style={{
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '16px',
        marginBottom: '12px',
        background: explanation.grounded ? '#f0fdf4' : '#fffbeb',
        boxShadow: 'var(--shadow)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>
          {language === 'en' ? 'Explanation' : 'स्पष्टीकरण'}
        </span>
        {explanation.grounded && (
          <span style={{ fontSize: '0.7rem', padding: '2px 6px', borderRadius: '4px', background: '#16a34a15', color: '#16a34a', border: '1px solid #16a34a40' }}>
            {language === 'en' ? 'Grounded' : 'सत्यापित'}
          </span>
        )}
        {!explanation.grounded && (
          <span style={{ fontSize: '0.7rem', padding: '2px 6px', borderRadius: '4px', background: '#d9770615', color: '#d97706', border: '1px solid #d9770640' }}>
            {language === 'en' ? 'Ungrounded' : 'असत्यापित'}
          </span>
        )}
      </div>
      <div
        style={{
          whiteSpace: 'pre-wrap',
          fontSize: '0.9rem',
          lineHeight: 1.6,
        }}
      >
        {explanation.text}
      </div>
      {explanation.citations.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary style={{ fontSize: '0.8rem', color: 'var(--muted)', cursor: 'pointer' }}>
            {language === 'en' ? 'Citations' : 'उद्धरण'}
          </summary>
          <ul style={{ marginTop: 8, paddingLeft: 20, fontSize: '0.8rem', color: 'var(--muted)' }}>
            {explanation.citations.map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </details>
      )}
    </div>
  )
}

function App() {
  const [text, setText] = useState('')
  const [language, setLanguage] = useState<Language>('en')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const analyze = useCallback(async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/analyze/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Analysis failed')
      }
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [text, language])

  const handleImage = useCallback(async (file: File) => {
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('language', language)
      const res = await fetch(`${API_BASE}/analyze/image`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Image analysis failed')
      }
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [language])

  return (
    <div style={{ minHeight: '100vh', padding: '24px 16px' }}>
      <header style={{ maxWidth: '900px', margin: '0 auto 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
          <img src="/medguard-logo.svg" alt="" width={40} height={40} />
          <div>
            <h1 style={{ fontSize: '1.75rem', fontWeight: 700 }}>MedGuard</h1>
            <p style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
              {language === 'en'
                ? 'Prescription safety AI — drug interactions, severity, and explanations'
                : 'प्रिस्क्रिप्शन सुरक्षा AI — दवा अंतरिक्रिया, गंभीरता, और स्पष्टीकरण'}
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            onClick={() => setLanguage('en')}
            style={{
              padding: '6px 14px', borderRadius: '9999px', fontSize: '0.85rem',
              fontWeight: 500, background: language === 'en' ? 'var(--primary)' : 'var(--bg)',
              color: language === 'en' ? 'white' : 'var(--text)', border: '1px solid var(--border)',
            }}
          >
            English
          </button>
          <button
            onClick={() => setLanguage('hi')}
            style={{
              padding: '6px 14px', borderRadius: '9999px', fontSize: '0.85rem',
              fontWeight: 500, background: language === 'hi' ? 'var(--primary)' : 'var(--bg)',
              color: language === 'hi' ? 'white' : 'var(--text)', border: '1px solid var(--border)',
            }}
          >
            हिंदी
          </button>
        </div>
      </header>

      <main style={{ maxWidth: '900px', margin: '0 auto' }}>
        <section
          style={{
            background: 'var(--card)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius)', padding: '20px', marginBottom: '24px',
            boxShadow: 'var(--shadow)',
          }}
        >
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 500 }}>
            {language === 'en' ? 'Prescription text' : 'प्रिस्क्रिप्शन टेक्स्ट'}
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={language === 'en'
              ? 'Paste prescription text here, e.g. "Tab Ecosprin AV 75mg, Tab Warfarin 5mg, Tab Metformin 500mg BD"'
              : 'यहाँ प्रिस्क्रिप्शन टेक्स्ट पेस्ट करें, जैसे "Tab Ecosprin AV 75mg, Tab Warfarin 5mg, Tab Metformin 500mg BD"'}
            rows={4}
            style={{
              width: '100%', padding: '12px', borderRadius: '8px',
              border: '1px solid var(--border)', fontFamily: 'inherit',
              resize: 'vertical', minHeight: '100px',
            }}
          />
          <div style={{ display: 'flex', gap: '12px', marginTop: '12px', flexWrap: 'wrap' }}>
            <button
              onClick={analyze}
              disabled={loading || !text.trim()}
              style={{
                padding: '10px 20px', borderRadius: '8px', fontSize: '0.95rem',
                fontWeight: 600, background: 'var(--primary)', color: 'white',
                opacity: loading || !text.trim() ? 0.6 : 1,
              }}
            >
              {loading ? (language === 'en' ? 'Analysing…' : 'विश्लेषण हो रहा है…') : (language === 'en' ? 'Analyze' : 'विश्लेषण करें')}
            </button>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && handleImage(e.target.files[0])} style={{ display: 'none' }} id="img-upload" />
              <button
                type="button"
                onClick={() => document.getElementById('img-upload')?.click()}
                disabled={loading}
                style={{
                  padding: '10px 20px', borderRadius: '8px', fontSize: '0.95rem',
                  fontWeight: 600, background: 'var(--bg)', color: 'var(--text)',
                  border: '1px solid var(--border)', opacity: loading ? 0.6 : 1,
                }}
              >
                {language === 'en' ? '📷 Upload prescription image' : '📷 प्रिस्क्रिप्शन छवि अपलोड करें'}
              </button>
            </label>
          </div>
        </section>

        {error && (
          <div
            style={{
              background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 'var(--radius)',
              padding: '16px', marginBottom: '24px', color: '#dc2626',
            }}
          >
            {error}
          </div>
        )}

        {result && (
          <>
            <section style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                  {language === 'en' ? 'Identified medicines' : 'पहचानी गई दवाइयाँ'}
                </h2>
                {result.drugs.length > 0 && (
                  <span style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                    {result.drugs.length} {language === 'en' ? 'medicines found' : 'दवाइयाँ मिलीं'}
                  </span>
                )}
              </div>
              {result.drugs.length === 0 ? (
                <p style={{ color: 'var(--muted)', padding: '16px', textAlign: 'center' }}>
                  {language === 'en' ? 'No medicines identified.' : 'कोई दवा पहचानी नहीं गई।'}
                </p>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ background: 'var(--bg)', borderBottom: '1px solid var(--border)' }}>
                        <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, fontSize: '0.85rem' }}>
                          {language === 'en' ? 'Medicine' : 'दवा'}
                        </th>
                        <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, fontSize: '0.85rem' }}>
                          {language === 'en' ? 'Dose' : 'खुराक'}
                        </th>
                        <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: 600, fontSize: '0.85rem' }}>
                          {language === 'en' ? 'Confidence' : 'विश्वास'}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.drugs.map((drug, i) => (
                        <DrugRow key={i} drug={drug} />
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>

            {result.abstention && (
              <section style={{ marginBottom: '24px' }}>
                <div
                  style={{
                    background: '#fffbeb', border: '1px solid #fcd34d', borderRadius: 'var(--radius)',
                    padding: '16px', marginBottom: '24px',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
                    <span style={{ fontSize: '1.25rem' }}>⚠️</span>
                    <div>
                      <strong style={{ color: '#92400e' }}>
                        {language === 'en' ? 'Could not give a definitive answer' : 'स्पष्ट उत्तर नहीं दे सके'}
                      </strong>
                      <p style={{ marginTop: 4, color: '#78350f' }}>{result.abstention.reason}</p>
                    </div>
                  </div>
                </div>
              </section>
            )}

            <section style={{ marginBottom: '24px' }}>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '16px' }}>
                {language === 'en' ? 'Interaction findings' : 'अंतरिक्रिया के परिणाम'}
              </h2>
              {result.findings.length === 0 ? (
                <div
                  style={{
                    background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 'var(--radius)',
                    padding: '24px', textAlign: 'center',
                  }}
                >
                  <span style={{ fontSize: '2rem' }}>✅</span>
                  <p style={{ marginTop: 8, color: '#166534', fontWeight: 500 }}>
                    {language === 'en'
                      ? 'No known interactions were found between these medicines.'
                      : 'इन दवाइयों के बीच कोई ज्ञात अंतरिक्रिया नहीं मिली।'}
                  </p>
                  <p style={{ marginTop: 4, color: '#166534', fontSize: '0.85rem' }}>
                    {language === 'en'
                      ? 'Still, always confirm with your doctor or pharmacist.'
                      : 'फिर भी, अपने डॉक्टर या फार्मेसिस्ट से पुष्टि करें।'}
                  </p>
                </div>
              ) : (
                <div>
                  {result.findings.map((f, i) => (
                    <FindingCard key={i} finding={f} language={language} index={i} />
                  ))}
                </div>
              )}
            </section>

            <section>
              <h2 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '16px' }}>
                {language === 'en' ? 'Explanations' : 'स्पष्टीकरण'}
              </h2>
              {result.explanations.map((exp, i) => (
                <ExplanationCard key={i} explanation={exp} language={language} />
              ))}
            </section>

            <footer style={{ marginTop: '32px', paddingTop: '16px', borderTop: '1px solid var(--border)', textAlign: 'center', color: 'var(--muted)', fontSize: '0.8rem' }}>
              <p>{language === 'en' ? 'Processed in' : 'प्रसंस्करण समय'} {result.processing_ms} ms</p>
              <p style={{ marginTop: 8 }}>
                {language === 'en'
                  ? '⚠️ Decision-support only. Not medical advice. Confirm with your doctor or pharmacist.'
                  : '⚠️ केवल सहायता जानकारी। चिकित्सा सलाह नहीं। अपने डॉक्टर या फार्मेसिस्ट से पुष्टि करें।'}
              </p>
            </footer>
          </>
        )}
      </main>
    </div>
  )
}

export default App