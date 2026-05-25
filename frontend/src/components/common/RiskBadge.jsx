import './RiskBadge.css';

export default function RiskBadge({ level, label, color, size = 'md' }) {
  if (level && !label) {
    const riskClass = level?.toLowerCase() || 'low';
    return (
      <span className={`risk-badge risk-badge-${riskClass} ${size === 'lg' ? 'risk-badge-lg' : ''}`}>
        <span className="risk-badge-dot" />
        {level}
      </span>
    );
  }

  const badgeLabel = label || level || 'Unknown';
  const badgeColor = color || '#94a3b8';

  return (
    <span 
      className={`risk-badge ${size === 'lg' ? 'risk-badge-lg' : ''}`}
      style={{
        background: `${badgeColor}15`,
        border: `1px solid ${badgeColor}30`,
        color: badgeColor
      }}
    >
      <span className="risk-badge-dot" style={{ backgroundColor: badgeColor }} />
      {badgeLabel}
    </span>
  );
}
