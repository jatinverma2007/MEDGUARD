import { useState, useCallback, useRef, useEffect } from 'react'

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

const SEVERITY_CONFIG: Record<string, { color: string; bg: string; label: { en: string; hi: string }; icon: string }> = {
  none: { color: '#10b981', bg: 'rgba(16, 185, 129, 0.1)', label: { en: 'No Interaction', hi: 'कोई अंतरिक्रिया नहीं' }, icon: '✅' },
  minor: { color: '#84cc16', bg: 'rgba(132, 204, 22, 0.1)', label: { en: 'Minor', hi: 'हल्का' }, icon: '🟢' },
  moderate: { color: '#f59e0b', bg: 'rgba(245, 158, 11, 0.1)', label: { en: 'Moderate', hi: 'मध्यम' }, icon: '🟡' },
  major: { color: '#ef4444', bg: 'rgba(239, 68, 68, 0.1)', label: { en: 'Major', hi: 'गंभीर' }, icon: '🔴' },
  contraindicated: { color: '#991b1b', bg: 'rgba(153, 27, 27, 0.1)', label: { en: 'Contraindicated', hi: 'वर्जित' }, icon: '⛔' },
  unknown: { color: '#6b7280', bg: 'rgba(107, 114, 128, 0.1)', label: { en: 'Unknown', hi: 'अज्ञात' }, icon: '❓' },
}

function SeverityBadge({ severity, language, large }: { severity: string; language: Language; large?: boolean }) {
  const config = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.unknown
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: large ? '8px 16px' : '4px 12px',
        borderRadius: '9999px',
        fontSize: large ? '0.95rem' : '0.75rem',
        fontWeight: 600,
        background: config.bg,
        color: config.color,
        border: `1.5px solid ${config.color}30`,
        letterSpacing: '0.01em',
      }}
    >
      <span>{config.icon}</span>
      {config.label[language]}
    </span>
  )
}

function ConfidenceBar({ value, color }: { value: number; color?: string }) {
  const percentage = Math.round(value * 100)
  const barColor = color || (percentage >= 80 ? '#10b981' : percentage >= 60 ? '#f59e0b' : '#ef4444')
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
      <div
        style={{
          flex: 1,
          height: '6px',
          borderRadius: '3px',
          background: 'rgba(0,0,0,0.06)',
          overflow: 'hidden',
          maxWidth: '100px',
        }}
      >
        <div
          style={{
            width: `${percentage}%`,
            height: '100%',
            borderRadius: '3px',
            background: `linear-gradient(90deg, ${barColor}aa, ${barColor})`,
            transition: 'width 0.8s cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        />
      </div>
      <span style={{ fontSize: '0.8rem', fontWeight: 600, color: barColor, minWidth: '36px' }}>{percentage}%</span>
    </div>
  )
}

function LoadingSpinner() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', padding: '40px 0' }}>
      <div
        style={{
          width: '48px',
          height: '48px',
          border: '3px solid rgba(0, 180, 216, 0.15)',
          borderTopColor: '#00b4d8',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite',
        }}
      />
      <span style={{ color: '#64748b', fontSize: '0.9rem', fontWeight: 500, animation: 'pulse 1.5s ease-in-out infinite' }}>
        Analyzing prescription...
      </span>
    </div>
  )
}

function DrugCard({ drug, index }: { drug: Drug; index: number }) {
  return (
    <div
      style={{
        background: '#ffffff',
        borderRadius: '12px',
        padding: '18px 20px',
        border: '1px solid #e2e8f0',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        animation: `fadeInUp 0.4s ease-out ${index * 0.05}s both`,
        cursor: 'default',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.1)'
        e.currentTarget.style.transform = 'translateY(-2px)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.06)'
        e.currentTarget.style.transform = 'translateY(0)'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '14px' }}>
        <div
          style={{
            width: '40px',
            height: '40px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, rgba(0,180,216,0.1), rgba(0,180,216,0.05))',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.2rem',
            flexShrink: 0,
          }}
        >
          💊
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 600, fontSize: '0.95rem', color: '#0f172a', marginBottom: '2px' }}>
            {drug.canonical_name || drug.normalized || drug.surface}
          </div>
          {drug.surface !== (drug.canonical_name || drug.normalized || drug.surface) && (
            <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginBottom: '4px' }}>
              as written: {drug.surface}
            </div>
          )}
          {drug.dose && (
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '4px',
                padding: '2px 8px',
                borderRadius: '6px',
                background: 'rgba(0,180,216,0.08)',
                color: '#0096b7',
                fontSize: '0.78rem',
                fontWeight: 500,
                marginTop: '4px',
              }}
            >
              {drug.dose}
            </div>
          )}
          {drug.confidence !== null && (
            <div style={{ marginTop: '8px' }}>
              <ConfidenceBar value={drug.confidence} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function FindingCard({ finding, language, index }: { finding: Finding; language: Language; index: number }) {
  const config = SEVERITY_CONFIG[finding.severity] || SEVERITY_CONFIG.unknown
  const [evidenceOpen, setEvidenceOpen] = useState(false)

  return (
    <div
      style={{
        background: '#ffffff',
        borderRadius: '12px',
        padding: '0',
        border: '1px solid #e2e8f0',
        boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
        overflow: 'hidden',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        animation: `fadeInUp 0.4s ease-out ${index * 0.08}s both`,
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.boxShadow = '0 8px 25px rgba(0,0,0,0.1)'
        e.currentTarget.style.transform = 'translateY(-2px)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.06)'
        e.currentTarget.style.transform = 'translateY(0)'
      }}
    >
      {/* Color-coded left border accent */}
      <div style={{ display: 'flex' }}>
        <div style={{ width: '4px', background: config.color, flexShrink: 0 }} />
        <div style={{ flex: 1, padding: '18px 20px' }}>
          {/* Header */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '14px', flexWrap: 'wrap' }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '6px 12px',
                borderRadius: '8px',
                background: 'rgba(10,22,40,0.04)',
                fontWeight: 600,
                fontSize: '0.9rem',
              }}
            >
              <span>💊</span>
              <span>{finding.drug_a}</span>
              <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>✕</span>
              <span>{finding.drug_b}</span>
            </div>
            <SeverityBadge severity={finding.severity} language={language} />
            <span
              style={{
                marginLeft: 'auto',
                fontSize: '0.72rem',
                color: '#94a3b8',
                padding: '3px 8px',
                borderRadius: '6px',
                background: 'rgba(0,0,0,0.03)',
                fontWeight: 500,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
              }}
            >
              {finding.source === 'kb' ? '📚 Knowledge Base' : finding.source === 'model' ? '🤖 AI Model' : '📚🤖 Both'}
            </span>
          </div>

          {/* Event type / Mechanism */}
          {finding.event_type && (
            <div style={{ marginBottom: '10px', display: 'flex', gap: '8px' }}>
              <span style={{ fontWeight: 600, color: '#64748b', fontSize: '0.82rem', minWidth: 'fit-content' }}>
                {language === 'en' ? 'Mechanism:' : 'कारण:'}
              </span>
              <span style={{ fontSize: '0.88rem', color: '#334155' }}>{finding.event_type}</span>
            </div>
          )}

          {/* Description */}
          {finding.description && (
            <div style={{ marginBottom: '10px', display: 'flex', gap: '8px' }}>
              <span style={{ fontWeight: 600, color: '#64748b', fontSize: '0.82rem', minWidth: 'fit-content' }}>
                {language === 'en' ? 'Description:' : 'विवरण:'}
              </span>
              <span style={{ fontSize: '0.88rem', color: '#334155', lineHeight: 1.6 }}>{finding.description}</span>
            </div>
          )}

          {/* Management */}
          {finding.management && (
            <div
              style={{
                marginBottom: '10px',
                padding: '10px 14px',
                borderRadius: '8px',
                background: 'rgba(16, 185, 129, 0.06)',
                border: '1px solid rgba(16, 185, 129, 0.15)',
              }}
            >
              <div style={{ fontWeight: 600, color: '#10b981', fontSize: '0.78rem', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                {language === 'en' ? '💡 Management Advice' : '💡 प्रबंधन सलाह'}
              </div>
              <span style={{ fontSize: '0.85rem', color: '#334155', lineHeight: 1.6 }}>{finding.management}</span>
            </div>
          )}

          {/* Confidence */}
          {finding.confidence && (
            <div style={{ marginBottom: '8px', fontSize: '0.78rem', color: '#64748b', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>{language === 'en' ? 'Calibrated confidence:' : 'कैलिब्रेटेड विश्वास:'}</span>
              <span style={{ fontWeight: 600 }}>{(finding.confidence.calibrated * 100).toFixed(1)}%</span>
              {finding.confidence.temperature !== 1.0 && (
                <span style={{ color: '#94a3b8' }}>(T = {finding.confidence.temperature.toFixed(3)})</span>
              )}
            </div>
          )}

          {/* Evidence toggle */}
          {finding.evidence.length > 0 && (
            <div style={{ marginTop: '10px' }}>
              <button
                onClick={() => setEvidenceOpen(!evidenceOpen)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  fontSize: '0.78rem',
                  color: '#00b4d8',
                  fontWeight: 500,
                  padding: '4px 0',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                }}
              >
                <span style={{ transition: 'transform 0.2s', transform: evidenceOpen ? 'rotate(90deg)' : 'rotate(0)' }}>▶</span>
                {language === 'en' ? `Evidence sources (${finding.evidence.length})` : `स्रोत (${finding.evidence.length})`}
              </button>
              {evidenceOpen && (
                <ul
                  style={{
                    marginTop: '8px',
                    paddingLeft: '20px',
                    fontSize: '0.8rem',
                    color: '#64748b',
                    animation: 'slideDown 0.2s ease-out',
                  }}
                >
                  {finding.evidence.map((e, i) => (
                    <li key={i} style={{ marginBottom: '4px' }}>{e}</li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function ExplanationCard({ explanation, language, index }: { explanation: Explanation; language: Language; index: number }) {
  const [citationsOpen, setCitationsOpen] = useState(false)

  return (
    <div
      style={{
        background: explanation.grounded
          ? 'linear-gradient(135deg, rgba(16,185,129,0.04), rgba(16,185,129,0.01))'
          : 'linear-gradient(135deg, rgba(245,158,11,0.04), rgba(245,158,11,0.01))',
        borderRadius: '12px',
        padding: '18px 20px',
        border: `1px solid ${explanation.grounded ? 'rgba(16,185,129,0.2)' : 'rgba(245,158,11,0.2)'}`,
        boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        animation: `fadeInUp 0.4s ease-out ${index * 0.08}s both`,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '12px' }}>
        <span style={{ fontSize: '1.1rem' }}>{explanation.grounded ? '🔬' : '💭'}</span>
        <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#0f172a' }}>
          {language === 'en' ? 'AI Explanation' : 'AI स्पष्टीकरण'}
        </span>
        <span
          style={{
            fontSize: '0.7rem',
            padding: '3px 10px',
            borderRadius: '9999px',
            fontWeight: 600,
            background: explanation.grounded ? 'rgba(16,185,129,0.1)' : 'rgba(245,158,11,0.1)',
            color: explanation.grounded ? '#10b981' : '#f59e0b',
            border: `1px solid ${explanation.grounded ? 'rgba(16,185,129,0.25)' : 'rgba(245,158,11,0.25)'}`,
          }}
        >
          {explanation.grounded
            ? (language === 'en' ? '✓ Grounded' : '✓ सत्यापित')
            : (language === 'en' ? '○ Ungrounded' : '○ असत्यापित')}
        </span>
      </div>
      <div
        style={{
          whiteSpace: 'pre-wrap',
          fontSize: '0.88rem',
          lineHeight: 1.7,
          color: '#334155',
        }}
      >
        {explanation.text}
      </div>
      {explanation.citations.length > 0 && (
        <div style={{ marginTop: '14px' }}>
          <button
            onClick={() => setCitationsOpen(!citationsOpen)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontSize: '0.78rem',
              color: '#00b4d8',
              fontWeight: 500,
              padding: '4px 0',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
            }}
          >
            <span style={{ transition: 'transform 0.2s', transform: citationsOpen ? 'rotate(90deg)' : 'rotate(0)' }}>▶</span>
            {language === 'en' ? `Citations (${explanation.citations.length})` : `उद्धरण (${explanation.citations.length})`}
          </button>
          {citationsOpen && (
            <ul
              style={{
                marginTop: '8px',
                paddingLeft: '20px',
                fontSize: '0.8rem',
                color: '#64748b',
                animation: 'slideDown 0.2s ease-out',
              }}
            >
              {explanation.citations.map((c, i) => (
                <li key={i} style={{ marginBottom: '4px' }}>{c}</li>
              ))}
            </ul>
          )}
        </div>
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
  const [activeTab, setActiveTab] = useState<'text' | 'image'>('text')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const resultsRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (result && resultsRef.current) {
      resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [result])

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file)
    const reader = new FileReader()
    reader.onload = (e) => setImagePreview(e.target?.result as string)
    reader.readAsDataURL(file)
  }, [])

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

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file)
    }
  }, [handleFileSelect])

  return (
    <div style={{ minHeight: '100vh', position: 'relative', overflow: 'hidden' }}>
      {/* Background floating medical cross watermark */}
      <div
        style={{
          position: 'fixed',
          top: '50%',
          right: '-80px',
          transform: 'translateY(-50%) rotate(15deg)',
          fontSize: '320px',
          opacity: 0.02,
          color: '#00b4d8',
          pointerEvents: 'none',
          zIndex: 0,
          fontWeight: 900,
          lineHeight: 1,
          userSelect: 'none',
        }}
      >
        ✚
      </div>
      <div
        style={{
          position: 'fixed',
          bottom: '-60px',
          left: '-40px',
          transform: 'rotate(-20deg)',
          fontSize: '240px',
          opacity: 0.015,
          color: '#0a1628',
          pointerEvents: 'none',
          zIndex: 0,
          fontWeight: 900,
          lineHeight: 1,
          userSelect: 'none',
        }}
      >
        ✚
      </div>

      {/* ==================== HEADER ==================== */}
      <header
        style={{
          background: 'linear-gradient(135deg, #0a1628 0%, #101d35 40%, #0f2847 70%, #0a3255 100%)',
          padding: '0',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: '0 4px 30px rgba(0,0,0,0.3)',
        }}
      >
        {/* Header decorative elements */}
        <div
          style={{
            position: 'absolute',
            top: '-50%',
            right: '-10%',
            width: '400px',
            height: '400px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(0,180,216,0.08) 0%, transparent 70%)',
            pointerEvents: 'none',
          }}
        />
        <div
          style={{
            position: 'absolute',
            bottom: '-30%',
            left: '10%',
            width: '300px',
            height: '300px',
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(0,180,216,0.05) 0%, transparent 70%)',
            pointerEvents: 'none',
          }}
        />

        <div
          style={{
            maxWidth: '1100px',
            margin: '0 auto',
            padding: '20px 24px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            position: 'relative',
            zIndex: 1,
            flexWrap: 'wrap',
            gap: '16px',
          }}
        >
          {/* Logo + title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <div
              style={{
                width: '48px',
                height: '48px',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(0,180,216,0.12)',
                border: '1px solid rgba(0,180,216,0.2)',
                boxShadow: '0 0 20px rgba(0,180,216,0.2)',
                animation: 'pulseGlow 3s ease-in-out infinite',
              }}
            >
              <img
                src="/medguard-logo.svg"
                alt="MedGuard"
                width={32}
                height={32}
                style={{ filter: 'drop-shadow(0 0 8px rgba(0,180,216,0.5))' }}
              />
            </div>
            <div>
              <h1
                style={{
                  fontSize: '1.5rem',
                  fontWeight: 800,
                  color: '#ffffff',
                  letterSpacing: '-0.02em',
                  lineHeight: 1.2,
                }}
              >
                Med
                <span style={{ color: '#00b4d8' }}>Guard</span>
              </h1>
              <p
                style={{
                  fontSize: '0.75rem',
                  color: 'rgba(148,163,184,0.9)',
                  fontWeight: 400,
                  letterSpacing: '0.02em',
                  marginTop: '2px',
                }}
              >
                {language === 'en' ? 'AI-Powered Prescription Safety System' : 'AI-संचालित प्रिस्क्रिप्शन सुरक्षा प्रणाली'}
              </p>
            </div>
          </div>

          {/* Right side: language toggle + status */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            {/* System Online indicator */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 12px',
                borderRadius: '9999px',
                background: 'rgba(16,185,129,0.1)',
                border: '1px solid rgba(16,185,129,0.2)',
              }}
            >
              <div
                style={{
                  width: '7px',
                  height: '7px',
                  borderRadius: '50%',
                  background: '#10b981',
                  boxShadow: '0 0 6px rgba(16,185,129,0.6)',
                  animation: 'pulse 2s ease-in-out infinite',
                }}
              />
              <span style={{ fontSize: '0.72rem', fontWeight: 500, color: '#10b981', letterSpacing: '0.03em' }}>
                {language === 'en' ? 'System Online' : 'सिस्टम ऑनलाइन'}
              </span>
            </div>

            {/* Language toggle */}
            <div
              style={{
                display: 'flex',
                borderRadius: '9999px',
                background: 'rgba(255,255,255,0.08)',
                border: '1px solid rgba(255,255,255,0.1)',
                padding: '3px',
              }}
            >
              <button
                onClick={() => setLanguage('en')}
                style={{
                  padding: '6px 16px',
                  borderRadius: '9999px',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  background: language === 'en'
                    ? 'linear-gradient(135deg, #00b4d8, #0096b7)'
                    : 'transparent',
                  color: language === 'en' ? '#ffffff' : 'rgba(255,255,255,0.6)',
                  border: 'none',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  cursor: 'pointer',
                }}
              >
                EN
              </button>
              <button
                onClick={() => setLanguage('hi')}
                style={{
                  padding: '6px 16px',
                  borderRadius: '9999px',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  background: language === 'hi'
                    ? 'linear-gradient(135deg, #00b4d8, #0096b7)'
                    : 'transparent',
                  color: language === 'hi' ? '#ffffff' : 'rgba(255,255,255,0.6)',
                  border: 'none',
                  transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                  cursor: 'pointer',
                }}
              >
                हिंदी
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ==================== MAIN ==================== */}
      <main
        style={{
          maxWidth: '1100px',
          margin: '0 auto',
          padding: '32px 24px 60px',
          position: 'relative',
          zIndex: 1,
        }}
      >
        {/* ==================== INPUT CARD ==================== */}
        <section
          style={{
            background: 'rgba(255,255,255,0.85)',
            backdropFilter: 'blur(20px)',
            WebkitBackdropFilter: 'blur(20px)',
            borderRadius: '20px',
            padding: '0',
            border: '1px solid rgba(255,255,255,0.6)',
            boxShadow: '0 10px 40px rgba(0,0,0,0.06), 0 2px 10px rgba(0,0,0,0.04)',
            overflow: 'hidden',
            marginBottom: '32px',
          }}
        >
          {/* Tab bar */}
          <div
            style={{
              display: 'flex',
              borderBottom: '1px solid #e2e8f0',
              background: 'rgba(248,250,252,0.6)',
            }}
          >
            <button
              onClick={() => setActiveTab('text')}
              style={{
                flex: 1,
                padding: '16px 20px',
                fontSize: '0.9rem',
                fontWeight: 600,
                color: activeTab === 'text' ? '#00b4d8' : '#64748b',
                background: 'transparent',
                border: 'none',
                borderBottom: activeTab === 'text' ? '2.5px solid #00b4d8' : '2.5px solid transparent',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
            >
              <span style={{ fontSize: '1.1rem' }}>📝</span>
              {language === 'en' ? 'Text Analysis' : 'टेक्स्ट विश्लेषण'}
            </button>
            <button
              onClick={() => setActiveTab('image')}
              style={{
                flex: 1,
                padding: '16px 20px',
                fontSize: '0.9rem',
                fontWeight: 600,
                color: activeTab === 'image' ? '#00b4d8' : '#64748b',
                background: 'transparent',
                border: 'none',
                borderBottom: activeTab === 'image' ? '2.5px solid #00b4d8' : '2.5px solid transparent',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
              }}
            >
              <span style={{ fontSize: '1.1rem' }}>📷</span>
              {language === 'en' ? 'Image Analysis' : 'छवि विश्लेषण'}
            </button>
          </div>

          {/* Tab content */}
          <div style={{ padding: '24px 28px 28px' }}>
            {activeTab === 'text' && (
              <div style={{ animation: 'fadeInUp 0.3s ease-out' }}>
                <label
                  style={{
                    display: 'block',
                    marginBottom: '10px',
                    fontWeight: 600,
                    fontSize: '0.88rem',
                    color: '#334155',
                  }}
                >
                  {language === 'en' ? 'Paste your prescription text below' : 'अपना प्रिस्क्रिप्शन टेक्स्ट नीचे पेस्ट करें'}
                </label>
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder={
                    language === 'en'
                      ? 'e.g. "Tab Ecosprin AV 75mg OD, Tab Warfarin 5mg OD, Tab Metformin 500mg BD, Cap Omeprazole 20mg OD"'
                      : 'जैसे "Tab Ecosprin AV 75mg OD, Tab Warfarin 5mg OD, Tab Metformin 500mg BD, Cap Omeprazole 20mg OD"'
                  }
                  rows={5}
                  style={{
                    width: '100%',
                    padding: '16px 18px',
                    borderRadius: '12px',
                    border: '1.5px solid #e2e8f0',
                    fontFamily: 'inherit',
                    fontSize: '0.92rem',
                    resize: 'vertical',
                    minHeight: '120px',
                    background: '#ffffff',
                    color: '#0f172a',
                    lineHeight: 1.6,
                    transition: 'all 0.3s ease',
                  }}
                />
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '16px' }}>
                  <button
                    onClick={analyze}
                    disabled={loading || !text.trim()}
                    style={{
                      padding: '12px 28px',
                      borderRadius: '12px',
                      fontSize: '0.92rem',
                      fontWeight: 600,
                      background: loading || !text.trim()
                        ? 'linear-gradient(135deg, #94a3b8, #94a3b8)'
                        : 'linear-gradient(135deg, #00b4d8, #0096b7)',
                      color: '#ffffff',
                      border: 'none',
                      cursor: loading || !text.trim() ? 'not-allowed' : 'pointer',
                      opacity: loading || !text.trim() ? 0.6 : 1,
                      boxShadow: loading || !text.trim()
                        ? 'none'
                        : '0 4px 15px rgba(0,180,216,0.3)',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      letterSpacing: '0.01em',
                    }}
                    onMouseEnter={(e) => {
                      if (!loading && text.trim()) {
                        e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,180,216,0.4)'
                        e.currentTarget.style.transform = 'translateY(-1px)'
                      }
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.boxShadow = loading || !text.trim() ? 'none' : '0 4px 15px rgba(0,180,216,0.3)'
                      e.currentTarget.style.transform = 'translateY(0)'
                    }}
                  >
                    {loading ? (
                      <>
                        <div
                          style={{
                            width: '16px',
                            height: '16px',
                            border: '2px solid rgba(255,255,255,0.3)',
                            borderTopColor: '#fff',
                            borderRadius: '50%',
                            animation: 'spin 0.6s linear infinite',
                          }}
                        />
                        {language === 'en' ? 'Analysing...' : 'विश्लेषण हो रहा है...'}
                      </>
                    ) : (
                      <>
                        <span>🔍</span>
                        {language === 'en' ? 'Analyze Prescription' : 'प्रिस्क्रिप्शन का विश्लेषण करें'}
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {activeTab === 'image' && (
              <div style={{ animation: 'fadeInUp 0.3s ease-out' }}>
                <input
                  type="file"
                  accept="image/*"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const file = e.target.files?.[0]
                    if (file) handleFileSelect(file)
                  }}
                />

                {/* Drop zone */}
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    border: `2px dashed ${isDragging ? '#00b4d8' : '#cbd5e1'}`,
                    borderRadius: '16px',
                    padding: '40px 24px',
                    textAlign: 'center',
                    cursor: 'pointer',
                    background: isDragging
                      ? 'rgba(0,180,216,0.05)'
                      : 'rgba(248,250,252,0.8)',
                    transition: 'all 0.3s ease',
                    marginBottom: '16px',
                  }}
                >
                  <div style={{ fontSize: '3rem', marginBottom: '12px', opacity: 0.7 }}>🏥</div>
                  <p style={{ fontWeight: 600, color: '#334155', fontSize: '0.95rem', marginBottom: '6px' }}>
                    {language === 'en' ? 'Drop prescription image here' : 'प्रिस्क्रिप्शन छवि यहाँ डालें'}
                  </p>
                  <p style={{ color: '#94a3b8', fontSize: '0.82rem' }}>
                    {language === 'en' ? 'or click to browse files • PNG, JPG, WEBP' : 'या फ़ाइलें ब्राउज़ करने के लिए क्लिक करें • PNG, JPG, WEBP'}
                  </p>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation()
                      fileInputRef.current?.click()
                    }}
                    style={{
                      marginTop: '14px',
                      padding: '8px 20px',
                      borderRadius: '10px',
                      fontSize: '0.85rem',
                      fontWeight: 500,
                      background: 'rgba(0,180,216,0.08)',
                      color: '#00b4d8',
                      border: '1px solid rgba(0,180,216,0.2)',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'rgba(0,180,216,0.15)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'rgba(0,180,216,0.08)'
                    }}
                  >
                    📁 {language === 'en' ? 'Browse Files' : 'फ़ाइलें ब्राउज़ करें'}
                  </button>
                </div>

                {/* Preview */}
                {selectedFile && imagePreview && (
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '14px',
                      padding: '14px 16px',
                      borderRadius: '12px',
                      background: 'rgba(0,180,216,0.04)',
                      border: '1px solid rgba(0,180,216,0.15)',
                      marginBottom: '16px',
                      animation: 'fadeInUp 0.3s ease-out',
                    }}
                  >
                    <img
                      src={imagePreview}
                      alt="Preview"
                      style={{
                        width: '56px',
                        height: '56px',
                        objectFit: 'cover',
                        borderRadius: '10px',
                        border: '2px solid rgba(0,180,216,0.2)',
                      }}
                    />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: '0.85rem', color: '#0f172a', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {selectedFile.name}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setSelectedFile(null)
                        setImagePreview(null)
                        if (fileInputRef.current) fileInputRef.current.value = ''
                      }}
                      style={{
                        width: '28px',
                        height: '28px',
                        borderRadius: '8px',
                        background: 'rgba(239,68,68,0.08)',
                        color: '#ef4444',
                        border: '1px solid rgba(239,68,68,0.15)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '0.8rem',
                        cursor: 'pointer',
                        flexShrink: 0,
                      }}
                    >
                      ✕
                    </button>
                  </div>
                )}

                {/* Upload button */}
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    onClick={() => selectedFile && handleImage(selectedFile)}
                    disabled={loading || !selectedFile}
                    style={{
                      padding: '12px 28px',
                      borderRadius: '12px',
                      fontSize: '0.92rem',
                      fontWeight: 600,
                      background: loading || !selectedFile
                        ? 'linear-gradient(135deg, #94a3b8, #94a3b8)'
                        : 'linear-gradient(135deg, #00b4d8, #0096b7)',
                      color: '#ffffff',
                      border: 'none',
                      cursor: loading || !selectedFile ? 'not-allowed' : 'pointer',
                      opacity: loading || !selectedFile ? 0.6 : 1,
                      boxShadow: loading || !selectedFile
                        ? 'none'
                        : '0 4px 15px rgba(0,180,216,0.3)',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}
                    onMouseEnter={(e) => {
                      if (!loading && selectedFile) {
                        e.currentTarget.style.boxShadow = '0 6px 20px rgba(0,180,216,0.4)'
                        e.currentTarget.style.transform = 'translateY(-1px)'
                      }
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.boxShadow = loading || !selectedFile ? 'none' : '0 4px 15px rgba(0,180,216,0.3)'
                      e.currentTarget.style.transform = 'translateY(0)'
                    }}
                  >
                    {loading ? (
                      <>
                        <div
                          style={{
                            width: '16px',
                            height: '16px',
                            border: '2px solid rgba(255,255,255,0.3)',
                            borderTopColor: '#fff',
                            borderRadius: '50%',
                            animation: 'spin 0.6s linear infinite',
                          }}
                        />
                        {language === 'en' ? 'Analysing...' : 'विश्लेषण हो रहा है...'}
                      </>
                    ) : (
                      <>
                        <span>📤</span>
                        {language === 'en' ? 'Upload & Analyze' : 'अपलोड और विश्लेषण करें'}
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Loading indicator */}
            {loading && <LoadingSpinner />}
          </div>
        </section>

        {/* ==================== ERROR ==================== */}
        {error && (
          <div
            style={{
              background: 'linear-gradient(135deg, rgba(239,68,68,0.06), rgba(239,68,68,0.02))',
              border: '1px solid rgba(239,68,68,0.2)',
              borderRadius: '14px',
              padding: '16px 20px',
              marginBottom: '24px',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              animation: 'fadeInUp 0.3s ease-out',
            }}
          >
            <span style={{ fontSize: '1.3rem' }}>❌</span>
            <div>
              <div style={{ fontWeight: 600, color: '#ef4444', fontSize: '0.88rem', marginBottom: '2px' }}>
                {language === 'en' ? 'Analysis Error' : 'विश्लेषण त्रुटि'}
              </div>
              <div style={{ color: '#991b1b', fontSize: '0.85rem' }}>{error}</div>
            </div>
          </div>
        )}

        {/* ==================== EMPTY STATE ==================== */}
        {!result && !loading && !error && (
          <div
            style={{
              textAlign: 'center',
              padding: '60px 24px',
              animation: 'fadeInUp 0.5s ease-out',
            }}
          >
            <div style={{ fontSize: '4rem', marginBottom: '16px', opacity: 0.3 }}>🩺</div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#64748b', marginBottom: '8px' }}>
              {language === 'en' ? 'Ready to analyze your prescription' : 'आपके प्रिस्क्रिप्शन का विश्लेषण करने के लिए तैयार'}
            </h3>
            <p style={{ fontSize: '0.85rem', color: '#94a3b8', maxWidth: '400px', margin: '0 auto' }}>
              {language === 'en'
                ? 'Enter prescription text or upload an image to check for drug interactions and safety information.'
                : 'दवा अंतरिक्रिया और सुरक्षा जानकारी जांचने के लिए प्रिस्क्रिप्शन टेक्स्ट दर्ज करें या एक छवि अपलोड करें।'}
            </p>
          </div>
        )}

        {/* ==================== RESULTS ==================== */}
        {result && (
          <div ref={resultsRef} style={{ animation: 'fadeInUp 0.5s ease-out' }}>
            {/* Overall Severity Banner */}
            <div
              style={{
                background: `linear-gradient(135deg, ${(SEVERITY_CONFIG[result.overall_severity] || SEVERITY_CONFIG.unknown).color}12, ${(SEVERITY_CONFIG[result.overall_severity] || SEVERITY_CONFIG.unknown).color}05)`,
                border: `1.5px solid ${(SEVERITY_CONFIG[result.overall_severity] || SEVERITY_CONFIG.unknown).color}25`,
                borderRadius: '16px',
                padding: '24px 28px',
                marginBottom: '28px',
                display: 'flex',
                alignItems: 'center',
                gap: '18px',
                boxShadow: `0 4px 20px ${(SEVERITY_CONFIG[result.overall_severity] || SEVERITY_CONFIG.unknown).color}10`,
              }}
            >
              <div
                style={{
                  width: '56px',
                  height: '56px',
                  borderRadius: '14px',
                  background: `${(SEVERITY_CONFIG[result.overall_severity] || SEVERITY_CONFIG.unknown).color}15`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.8rem',
                  flexShrink: 0,
                }}
              >
                {(SEVERITY_CONFIG[result.overall_severity] || SEVERITY_CONFIG.unknown).icon}
              </div>
              <div>
                <div style={{ fontSize: '0.78rem', fontWeight: 500, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '4px' }}>
                  {language === 'en' ? 'Overall Assessment' : 'समग्र मूल्यांकन'}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '1.3rem', fontWeight: 700, color: (SEVERITY_CONFIG[result.overall_severity] || SEVERITY_CONFIG.unknown).color }}>
                    {(SEVERITY_CONFIG[result.overall_severity] || SEVERITY_CONFIG.unknown).label[language]}
                  </span>
                  <span
                    style={{
                      padding: '3px 10px',
                      borderRadius: '8px',
                      background: 'rgba(0,0,0,0.04)',
                      fontSize: '0.78rem',
                      color: '#64748b',
                      fontWeight: 500,
                    }}
                  >
                    {result.drugs.length} {language === 'en' ? 'medicines' : 'दवाइयाँ'} • {result.findings.length} {language === 'en' ? 'interactions' : 'अंतरिक्रिया'}
                  </span>
                </div>
              </div>
            </div>

            {/* Abstention Warning */}
            {result.abstention && (
              <div
                style={{
                  background: 'linear-gradient(135deg, rgba(245,158,11,0.08), rgba(245,158,11,0.02))',
                  border: '1.5px solid rgba(245,158,11,0.25)',
                  borderRadius: '14px',
                  padding: '18px 22px',
                  marginBottom: '28px',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '14px',
                  animation: 'fadeInUp 0.4s ease-out',
                }}
              >
                <div
                  style={{
                    width: '40px',
                    height: '40px',
                    borderRadius: '10px',
                    background: 'rgba(245,158,11,0.12)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '1.3rem',
                    flexShrink: 0,
                  }}
                >
                  ⚠️
                </div>
                <div>
                  <div style={{ fontWeight: 700, color: '#92400e', fontSize: '0.92rem', marginBottom: '4px' }}>
                    {language === 'en' ? 'Abstention Warning' : 'अनिर्णय चेतावनी'}
                  </div>
                  <p style={{ color: '#78350f', fontSize: '0.85rem', lineHeight: 1.6 }}>
                    {result.abstention.reason}
                  </p>
                  {result.abstention.confidence !== null && (
                    <div style={{ marginTop: '8px', fontSize: '0.78rem', color: '#92400e' }}>
                      {language === 'en' ? 'Confidence:' : 'विश्वास:'} {(result.abstention.confidence * 100).toFixed(1)}%
                      {result.abstention.threshold !== null && (
                        <span> (threshold: {(result.abstention.threshold * 100).toFixed(1)}%)</span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* ==================== IDENTIFIED MEDICINES ==================== */}
            <section style={{ marginBottom: '32px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '1.2rem' }}>💊</span>
                  <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>
                    {language === 'en' ? 'Identified Medicines' : 'पहचानी गई दवाइयाँ'}
                  </h2>
                </div>
                {result.drugs.length > 0 && (
                  <span
                    style={{
                      padding: '4px 12px',
                      borderRadius: '9999px',
                      background: 'rgba(0,180,216,0.08)',
                      color: '#0096b7',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                    }}
                  >
                    {result.drugs.length} {language === 'en' ? 'found' : 'मिलीं'}
                  </span>
                )}
              </div>

              {result.drugs.length === 0 ? (
                <div
                  style={{
                    padding: '32px',
                    textAlign: 'center',
                    color: '#94a3b8',
                    background: 'rgba(248,250,252,0.6)',
                    borderRadius: '14px',
                    border: '1px solid #e2e8f0',
                  }}
                >
                  <div style={{ fontSize: '2rem', marginBottom: '8px', opacity: 0.4 }}>💊</div>
                  <p style={{ fontSize: '0.88rem' }}>
                    {language === 'en' ? 'No medicines identified.' : 'कोई दवा पहचानी नहीं गई।'}
                  </p>
                </div>
              ) : (
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
                    gap: '12px',
                  }}
                >
                  {result.drugs.map((drug, i) => (
                    <DrugCard key={i} drug={drug} index={i} />
                  ))}
                </div>
              )}
            </section>

            {/* ==================== DRUG INTERACTIONS ==================== */}
            <section style={{ marginBottom: '32px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '1.2rem' }}>⚡</span>
                  <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>
                    {language === 'en' ? 'Drug Interactions' : 'दवा अंतरिक्रिया'}
                  </h2>
                </div>
                {result.findings.length > 0 && (
                  <span
                    style={{
                      padding: '4px 12px',
                      borderRadius: '9999px',
                      background: result.findings.some(f => f.severity === 'major' || f.severity === 'contraindicated')
                        ? 'rgba(239,68,68,0.08)'
                        : 'rgba(245,158,11,0.08)',
                      color: result.findings.some(f => f.severity === 'major' || f.severity === 'contraindicated')
                        ? '#ef4444'
                        : '#f59e0b',
                      fontSize: '0.78rem',
                      fontWeight: 600,
                    }}
                  >
                    {result.findings.length} {language === 'en' ? 'found' : 'मिलीं'}
                  </span>
                )}
              </div>

              {result.findings.length === 0 ? (
                <div
                  style={{
                    background: 'linear-gradient(135deg, rgba(16,185,129,0.06), rgba(16,185,129,0.01))',
                    border: '1.5px solid rgba(16,185,129,0.2)',
                    borderRadius: '14px',
                    padding: '32px',
                    textAlign: 'center',
                  }}
                >
                  <div style={{ fontSize: '2.5rem', marginBottom: '10px' }}>✅</div>
                  <p style={{ color: '#166534', fontWeight: 600, fontSize: '1rem', marginBottom: '6px' }}>
                    {language === 'en'
                      ? 'No known interactions found'
                      : 'कोई ज्ञात अंतरिक्रिया नहीं मिली'}
                  </p>
                  <p style={{ color: '#15803d', fontSize: '0.82rem' }}>
                    {language === 'en'
                      ? 'Still, always confirm with your doctor or pharmacist.'
                      : 'फिर भी, अपने डॉक्टर या फार्मेसिस्ट से पुष्टि करें।'}
                  </p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {result.findings.map((f, i) => (
                    <FindingCard key={i} finding={f} language={language} index={i} />
                  ))}
                </div>
              )}
            </section>

            {/* ==================== AI EXPLANATIONS ==================== */}
            {result.explanations.length > 0 && (
              <section style={{ marginBottom: '32px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '16px' }}>
                  <span style={{ fontSize: '1.2rem' }}>🧠</span>
                  <h2 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0f172a' }}>
                    {language === 'en' ? 'AI Explanations' : 'AI स्पष्टीकरण'}
                  </h2>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {result.explanations.map((exp, i) => (
                    <ExplanationCard key={i} explanation={exp} language={language} index={i} />
                  ))}
                </div>
              </section>
            )}

            {/* ==================== FOOTER ==================== */}
            <footer
              style={{
                marginTop: '40px',
                paddingTop: '20px',
                borderTop: '1px solid #e2e8f0',
                textAlign: 'center',
              }}
            >
              {result.processing_ms !== null && (
                <p
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    padding: '6px 14px',
                    borderRadius: '9999px',
                    background: 'rgba(0,180,216,0.06)',
                    color: '#0096b7',
                    fontSize: '0.78rem',
                    fontWeight: 500,
                    marginBottom: '12px',
                  }}
                >
                  ⚡ {language === 'en' ? 'Processed in' : 'प्रसंस्करण समय'} {result.processing_ms}ms
                </p>
              )}
              <p
                style={{
                  color: '#94a3b8',
                  fontSize: '0.78rem',
                  maxWidth: '500px',
                  margin: '8px auto 0',
                  lineHeight: 1.5,
                }}
              >
                {language === 'en'
                  ? '⚠️ Decision-support only. Not medical advice. Always confirm with your doctor or pharmacist.'
                  : '⚠️ केवल सहायता जानकारी। चिकित्सा सलाह नहीं। अपने डॉक्टर या फार्मेसिस्ट से पुष्टि करें।'}
              </p>
              <p
                style={{
                  color: '#cbd5e1',
                  fontSize: '0.72rem',
                  marginTop: '12px',
                }}
              >
                Powered by MedGuard AI Engine
              </p>
            </footer>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
