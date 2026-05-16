import { useLocation, useNavigate } from 'react-router-dom';
import { Home, RotateCcw, CheckCircle, ShieldCheck, AlertTriangle, AlertOctagon, Phone, Lightbulb } from 'lucide-react';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import RiskBadge from '../../components/common/RiskBadge';
import { RISK_RECOMMENDATIONS } from '../../data/uiConfig';
import './ResultPage.css';

export default function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { student, result } = location.state || {};

  if (!student || !result) {
    return (
      <div className="result-page">
        <div className="bg-pattern" />
        <div className="result-container" style={{ textAlign: 'center' }}>
          <h2>No results to display</h2>
          <p style={{ color: 'var(--text-secondary)', margin: 'var(--space-4) 0' }}>
            Please complete the screening questionnaire first.
          </p>
          <Button variant="primary" onClick={() => navigate('/')}>
            Start Over
          </Button>
        </div>
      </div>
    );
  }

  const { risk_level, probability, recommended_action } = result;
  const riskKey = risk_level;
  const riskConfig = RISK_RECOMMENDATIONS[riskKey];
  const riskClass = risk_level.toLowerCase();
  const probabilityScore = probability?.depressed || 0;
  const percentage = Math.round(probabilityScore * 100);
  const gaugeRotation = `${(probabilityScore * 360) - 90}deg`;

  const riskIcons = {
    Low: ShieldCheck,
    Moderate: AlertTriangle,
    High: AlertOctagon
  };
  const RiskIcon = riskIcons[riskKey];

  return (
    <div className="result-page">
      <div className="bg-pattern" />

      <div className="result-container">
        {/* Progress Complete */}
        <div className="result-progress">
          <div className="progress-steps" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
            <div className="progress-step done">
              <span className="progress-step-dot"><CheckCircle size={14} /></span>
              <span>Verify</span>
            </div>
            <span className="progress-step-line done" />
            <div className="progress-step done">
              <span className="progress-step-dot"><CheckCircle size={14} /></span>
              <span>Questionnaire</span>
            </div>
            <span className="progress-step-line done" />
            <div className="progress-step active">
              <span className="progress-step-dot">3</span>
              <span>Result</span>
            </div>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: '100%' }} />
          </div>
        </div>

        {/* Score Card */}
        <Card elevated className={`score-card glow-${riskClass}`}>
          <div className="score-gauge">
            <div className="score-gauge-ring">
              <div className="score-gauge-bg" />
              <div className="score-gauge-fill" style={{ '--gauge-rotation': gaugeRotation }} />
              <div className="score-gauge-inner">
                <span className={`score-value risk-${riskClass}`}>{percentage}%</span>
                <span className="score-label">Risk Score</span>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
            {RiskIcon && <RiskIcon size={24} className={`risk-${riskClass}`} />}
            <span className={`score-risk-level risk-${riskClass}`}>{riskConfig.title}</span>
          </div>

          <RiskBadge level={risk_level} size="lg" />

          <p className="score-message">{riskConfig.message}</p>
        </Card>

        {/* Recommendations */}
        <Card elevated className="recommendations">
          <div className="recommendations-title">
            <Lightbulb size={18} style={{ color: 'var(--accent-warning)' }} />
            Recommended Actions
          </div>
          <ul className="recommendations-list">
            {riskConfig.actions.map((action, i) => (
              <li key={i}>
                <span className="recommendation-icon" style={{
                  background: `${riskConfig.color}15`,
                  color: riskConfig.color
                }}>
                  <CheckCircle size={12} />
                </span>
                {action}
              </li>
            ))}
          </ul>
        </Card>

        {/* Helpline — shown for Moderate & High */}
        {(risk_level === 'High' || risk_level === 'Moderate') && (
          <div className="helpline-banner">
            <div className="helpline-title">
              <Phone size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
              Need to talk? Help is available 24/7
            </div>
            <div className="helpline-number">1800-599-0019</div>
            <div className="helpline-description">KIRAN Mental Health Helpline (Free, Confidential)</div>
          </div>
        )}

        {/* Actions */}
        <div className="result-actions">
          <Button variant="secondary" icon={RotateCcw} onClick={() => navigate('/')}>
            New Screening
          </Button>
          <Button variant="primary" icon={Home} onClick={() => navigate('/')}>
            Back to Home
          </Button>
        </div>
      </div>
    </div>
  );
}
