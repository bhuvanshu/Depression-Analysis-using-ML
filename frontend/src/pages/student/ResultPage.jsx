import { useLocation, useNavigate } from 'react-router-dom';
import { Home, RotateCcw, CheckCircle, ShieldCheck, AlertTriangle, AlertOctagon, Phone, Lightbulb } from 'lucide-react';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import RiskBadge from '../../components/common/RiskBadge';
import { RISK_RECOMMENDATIONS } from '../../data/uiConfig';
import { getSeverityInterpretation } from '../../utils/helpers';
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

  const { risk_level, institutional_priority, severity_interpretation, probability, recommended_action } = result;
  const riskKey = institutional_priority?.tier || risk_level || 'Low';
  const riskConfig = RISK_RECOMMENDATIONS[riskKey];
  const riskClass = riskKey.toLowerCase();
  const probabilityScore = probability?.depressed || 0;
  const percentage = Math.round(probabilityScore * 100);
  const gaugeRotation = `${(probabilityScore * 360) - 90}deg`;
  const severity = severity_interpretation || getSeverityInterpretation(probabilityScore);

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

          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
            {RiskIcon && <RiskIcon size={24} style={{ color: severity.color }} />}
            <span className="score-risk-level" style={{ color: severity.color, fontWeight: '700' }}>
               {severity.level}
            </span>
          </div>

          <p className="score-message">{severity.meaning}</p>
        </Card>

        {/* Recommendations */}
        <Card elevated className="recommendations">
          <div className="recommendations-title">
            <Lightbulb size={18} style={{ color: 'var(--accent-warning)' }} />
            Recommended Actions
          </div>
          
          {recommended_action && (
            <div className="ml-action-triage" style={{ 
              marginBottom: '1rem', 
              padding: '0.85rem', 
              background: 'rgba(255, 255, 255, 0.03)', 
              borderRadius: '6px', 
              borderLeft: `4px solid ${severity.color}` 
            }}>
              <strong style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>AI Triage Directive:</strong>
              <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.95rem', color: 'var(--text-primary)' }}>{recommended_action}</p>
            </div>
          )}
          
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
        {(riskKey === 'High' || riskKey === 'Moderate') && (
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
