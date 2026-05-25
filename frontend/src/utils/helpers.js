export const getInitials = (name) =>
  name ? name.split(' ').map(n => n[0]).join('').toUpperCase() : '?';

export const getSeverityInterpretation = (score) => {
  if (score > 0.85) {
    return {
      level: "High Risk Tendency",
      color: "#dc2626",
      meaning: "High probability of similarity to depressive-class patterns"
    };
  } else if (score > 0.60) {
    return {
      level: "Elevated Tendency",
      color: "#ea580c",
      meaning: "Moderate-to-high similarity to depressive-class patterns"
    };
  } else if (score > 0.35) {
    return {
      level: "Mild Tendency",
      color: "#f59e0b",
      meaning: "Some similarity to depressive-class patterns detected"
    };
  } else {
    return {
      level: "Minimal Tendency",
      color: "#22c55e",
      meaning: "Low similarity to depressive-class patterns"
    };
  }
};

export const getInstitutionalPriority = (riskLevel) => {
  const level = riskLevel || 'Low';
  const tierMap = {
    High: {
      tier: "High",
      color: "#ef4444",
      percentile: "Top 25%",
      action: "Requires priority attention and further evaluation"
    },
    Moderate: {
      tier: "Moderate",
      color: "#f59e0b",
      percentile: "Middle 50%",
      action: "Suggests monitoring and supportive interventions"
    },
    Low: {
      tier: "Low",
      color: "#22c55e",
      percentile: "Bottom 25%",
      action: "Indicates general awareness level"
    }
  };
  return tierMap[level] || tierMap.Low;
};
