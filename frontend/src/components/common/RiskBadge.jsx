import './RiskBadge.css';

export default function RiskBadge({ level, size = 'md' }) {
  const riskClass = level?.toLowerCase() || 'low';

  return (
    <span className={`risk-badge risk-badge-${riskClass} ${size === 'lg' ? 'risk-badge-lg' : ''}`}>
      <span className="risk-badge-dot" />
      {level}
    </span>
  );
}
