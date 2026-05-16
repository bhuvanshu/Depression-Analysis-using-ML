/* UI configuration for the Depression Analysis System frontend */


export const QUESTIONNAIRE_CONFIG = {
 _version : '1.0.1',
  academic_pressure: {
    label: "Academic Pressure",
    description: "How much academic pressure do you experience?",
    type: "slider",
    min: 0,
    max: 5,
    labels: { 0: "None", 1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Extreme" }
  },
  financial_stress: {
    label: "Financial Stress",
    description: "Rate your level of financial stress",
    type: "slider",
    min: 0,
    max: 5,
    labels: { 0: "None", 1: "Low", 2: "Mild", 3: "Moderate", 4: "High", 5: "Extreme" }
  },
  study_satisfaction: {
    label: "Study Satisfaction",
    description: "How satisfied are you with your studies?",
    type: "slider",
    min: 0,
    max: 5,
    labels: { 0: "Very Low", 1: "Low", 2: "Below Avg", 3: "Average", 4: "Good", 5: "Excellent" }
  },
  sleep_duration: {
    label: "Sleep Duration",
    description: "How many hours do you typically sleep?",
    type: "select",
    options: [
      { value: 1, label: "Less than 5 hours" },
      { value: 2, label: "5–6 hours" },
      { value: 3, label: "7–8 hours" },
      { value: 4, label: "More than 8 hours" }
    ]
  },
  work_study_hours: {
    label: "Daily Work/Study Hours",
    description: "Hours spent on work or study per day",
    type: "slider",
    min: 0,
    max: 5,
    labels: { 0: "0–2h", 1: "2–4h", 2: "4–6h", 3: "6–8h", 4: "8–10h", 5: "10–12h" }
  },
  suicidal_thoughts: {
    label: "Have you experienced suicidal thoughts?",
    description: "This is a safe, confidential screening question",
    type: "toggle",
    options: [
      { value: 0, label: "No" },
      { value: 1, label: "Yes" }
    ]
  },
  family_history: {
    label: "Family History of Mental Illness",
    description: "Is there a history of mental illness in your family?",
    type: "toggle",
    options: [
      { value: 0, label: "No" },
      { value: 1, label: "Yes" }
    ]
  }
};

export const RISK_RECOMMENDATIONS = {
  Low: {
    title: "Low Risk",
    color: "var(--accent-success)",
    glow: "var(--accent-success-glow)",
    icon: "shield-check",
    message: "Your responses indicate generally healthy mental well-being.",
    actions: [
      "Continue maintaining your current lifestyle habits",
      "Stay connected with friends and family",
      "Practice regular physical activity",
      "Access campus wellness resources for general well-being"
    ]
  },
  Moderate: {
    title: "Moderate Risk",
    color: "var(--accent-warning)",
    glow: "var(--accent-warning-glow)",
    icon: "alert-triangle",
    message: "Your responses suggest some areas that may benefit from attention.",
    actions: [
      "Consider scheduling a counseling session",
      "Join stress management workshops offered on campus",
      "Establish a regular sleep schedule",
      "Talk to a trusted friend, family member, or mentor",
      "Monitor your stress levels over the coming weeks"
    ]
  },
  High: {
    title: "High Risk — Please Seek Support",
    color: "var(--accent-danger)",
    glow: "var(--accent-danger-glow)",
    icon: "alert-octagon",
    message: "Your responses indicate significant distress. You are not alone — help is available.",
    actions: [
      "Please reach out to campus counseling services immediately",
      "Contact a mental health professional for evaluation",
      "If in crisis, call KIRAN helpline: 1800-599-0019 (24/7, free)",
      "Speak with someone you trust about how you're feeling",
      "Remember: seeking help is a sign of strength"
    ]
  }
};

