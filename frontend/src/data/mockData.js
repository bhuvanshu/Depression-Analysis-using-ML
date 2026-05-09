/* Mock data for the Depression Analysis System frontend */

export const MOCK_STUDENTS = [
  {
    student_id: 1,
    college_id: 1,
    enrollment_id: "BT21CSE001",
    name: "Arjun Mehta",
    age: 21,
    gender: "Male",
    department: "Computer Science",
    degree_group: "Undergraduate",
    cgpa: 8.2,
    created_at: "2026-01-15"
  },
  {
    student_id: 2,
    college_id: 1,
    enrollment_id: "BT21ECE045",
    name: "Priya Sharma",
    age: 22,
    gender: "Female",
    department: "Electronics",
    degree_group: "Undergraduate",
    cgpa: 7.5,
    created_at: "2026-01-15"
  },
  {
    student_id: 3,
    college_id: 1,
    enrollment_id: "MT22ME012",
    name: "Rahul Verma",
    age: 24,
    gender: "Male",
    department: "Mechanical",
    degree_group: "Postgraduate",
    cgpa: 8.8,
    created_at: "2026-02-01"
  },
  {
    student_id: 4,
    college_id: 1,
    enrollment_id: "BT20IT078",
    name: "Sneha Patel",
    age: 22,
    gender: "Female",
    department: "Information Technology",
    degree_group: "Undergraduate",
    cgpa: 6.9,
    created_at: "2026-01-20"
  },
  {
    student_id: 5,
    college_id: 1,
    enrollment_id: "PHD23CS003",
    name: "Vikram Singh",
    age: 28,
    gender: "Male",
    department: "Computer Science",
    degree_group: "Doctorate",
    cgpa: 9.1,
    created_at: "2026-03-01"
  },
  {
    student_id: 6,
    college_id: 1,
    enrollment_id: "BT22CV034",
    name: "Ananya Reddy",
    age: 20,
    gender: "Female",
    department: "Civil Engineering",
    degree_group: "Undergraduate",
    cgpa: 7.8,
    created_at: "2026-01-18"
  },
  {
    student_id: 7,
    college_id: 1,
    enrollment_id: "MT23EE009",
    name: "Karthik Nair",
    age: 25,
    gender: "Male",
    department: "Electrical",
    degree_group: "Postgraduate",
    cgpa: 8.5,
    created_at: "2026-02-10"
  },
  {
    student_id: 8,
    college_id: 1,
    enrollment_id: "BT21CH056",
    name: "Divya Joshi",
    age: 21,
    gender: "Female",
    department: "Chemical Engineering",
    degree_group: "Undergraduate",
    cgpa: 7.1,
    created_at: "2026-01-22"
  }
];

export const MOCK_RESPONSES = [
  {
    response_id: 1,
    student_id: 1,
    academic_pressure: 4,
    financial_stress: 3,
    study_satisfaction: 2,
    sleep_duration: 2,
    work_study_hours: 4,
    suicidal_thoughts: 0,
    family_history: 0,
    cgpa: 8.2,
    other_factors: "",
    submitted_at: "2026-04-10T10:30:00"
  },
  {
    response_id: 2,
    student_id: 2,
    academic_pressure: 3,
    financial_stress: 4,
    study_satisfaction: 3,
    sleep_duration: 3,
    work_study_hours: 3,
    suicidal_thoughts: 0,
    family_history: 1,
    cgpa: 7.5,
    other_factors: "Relationship issues",
    submitted_at: "2026-04-11T14:20:00"
  },
  {
    response_id: 3,
    student_id: 3,
    academic_pressure: 5,
    financial_stress: 5,
    study_satisfaction: 1,
    sleep_duration: 1,
    work_study_hours: 5,
    suicidal_thoughts: 1,
    family_history: 1,
    cgpa: 8.8,
    other_factors: "Research deadline pressure, isolation",
    submitted_at: "2026-04-12T09:15:00"
  },
  {
    response_id: 4,
    student_id: 4,
    academic_pressure: 2,
    financial_stress: 2,
    study_satisfaction: 4,
    sleep_duration: 3,
    work_study_hours: 2,
    suicidal_thoughts: 0,
    family_history: 0,
    cgpa: 6.9,
    other_factors: "",
    submitted_at: "2026-04-13T16:45:00"
  },
  {
    response_id: 5,
    student_id: 5,
    academic_pressure: 5,
    financial_stress: 4,
    study_satisfaction: 2,
    sleep_duration: 1,
    work_study_hours: 5,
    suicidal_thoughts: 1,
    family_history: 0,
    cgpa: 9.1,
    other_factors: "Publication pressure, advisor conflict",
    submitted_at: "2026-04-14T11:00:00"
  },
  {
    response_id: 6,
    student_id: 6,
    academic_pressure: 2,
    financial_stress: 1,
    study_satisfaction: 4,
    sleep_duration: 3,
    work_study_hours: 2,
    suicidal_thoughts: 0,
    family_history: 0,
    cgpa: 7.8,
    other_factors: "",
    submitted_at: "2026-04-15T13:30:00"
  },
  {
    response_id: 7,
    student_id: 7,
    academic_pressure: 4,
    financial_stress: 3,
    study_satisfaction: 2,
    sleep_duration: 2,
    work_study_hours: 4,
    suicidal_thoughts: 0,
    family_history: 1,
    cgpa: 8.5,
    other_factors: "Sleep disorder",
    submitted_at: "2026-04-16T10:00:00"
  },
  {
    response_id: 8,
    student_id: 8,
    academic_pressure: 3,
    financial_stress: 4,
    study_satisfaction: 3,
    sleep_duration: 2,
    work_study_hours: 3,
    suicidal_thoughts: 0,
    family_history: 0,
    cgpa: 7.1,
    other_factors: "Family pressure",
    submitted_at: "2026-04-17T15:20:00"
  }
];

export const MOCK_RESULTS = [
  {
    result_id: 1,
    response_id: 1,
    probability_score: 0.62,
    risk_level: "Moderate",
    recommendation: "Consider regular counseling sessions and stress management workshops. Monitor academic workload and ensure adequate rest.",
    predicted_at: "2026-04-10T10:30:05"
  },
  {
    result_id: 2,
    response_id: 2,
    probability_score: 0.71,
    risk_level: "Moderate",
    recommendation: "Family history increases risk factor. Recommend periodic psychological check-ins and peer support group participation.",
    predicted_at: "2026-04-11T14:20:03"
  },
  {
    result_id: 3,
    response_id: 3,
    probability_score: 0.92,
    risk_level: "High",
    recommendation: "Urgent: Schedule immediate counseling session. High academic pressure combined with suicidal ideation requires priority intervention. Connect with campus mental health services.",
    predicted_at: "2026-04-12T09:15:04"
  },
  {
    result_id: 4,
    response_id: 4,
    probability_score: 0.18,
    risk_level: "Low",
    recommendation: "Continue maintaining healthy lifestyle. General wellness resources and awareness programs recommended.",
    predicted_at: "2026-04-13T16:45:02"
  },
  {
    result_id: 5,
    response_id: 5,
    probability_score: 0.87,
    risk_level: "High",
    recommendation: "Urgent: Doctoral research stress combined with sleep deprivation and suicidal thoughts. Immediate referral to professional psychological support required.",
    predicted_at: "2026-04-14T11:00:06"
  },
  {
    result_id: 6,
    response_id: 6,
    probability_score: 0.12,
    risk_level: "Low",
    recommendation: "Student shows healthy indicators across all parameters. Continue regular check-ins as part of general wellness program.",
    predicted_at: "2026-04-15T13:30:02"
  },
  {
    result_id: 7,
    response_id: 7,
    probability_score: 0.68,
    risk_level: "Moderate",
    recommendation: "Family history of mental illness noted. Suggest ongoing counseling and sleep hygiene improvement program. Monitor academic stress levels.",
    predicted_at: "2026-04-16T10:00:03"
  },
  {
    result_id: 8,
    response_id: 8,
    probability_score: 0.55,
    risk_level: "Moderate",
    recommendation: "Financial stress and family pressure are contributing factors. Recommend financial counseling services and peer support engagement.",
    predicted_at: "2026-04-17T15:20:04"
  }
];

export const DEPARTMENT_STATS = [
  { department: "Computer Science", total: 45, high: 8, moderate: 15, low: 22 },
  { department: "Electronics", total: 38, high: 5, moderate: 12, low: 21 },
  { department: "Mechanical", total: 32, high: 6, moderate: 10, low: 16 },
  { department: "Information Technology", total: 40, high: 7, moderate: 14, low: 19 },
  { department: "Civil Engineering", total: 28, high: 3, moderate: 9, low: 16 },
  { department: "Electrical", total: 30, high: 4, moderate: 11, low: 15 },
  { department: "Chemical Engineering", total: 22, high: 3, moderate: 7, low: 12 }
];

export const MONTHLY_TRENDS = [
  { month: "Nov", screenings: 45, highRisk: 8 },
  { month: "Dec", screenings: 62, highRisk: 12 },
  { month: "Jan", screenings: 78, highRisk: 15 },
  { month: "Feb", screenings: 95, highRisk: 18 },
  { month: "Mar", screenings: 110, highRisk: 22 },
  { month: "Apr", screenings: 134, highRisk: 28 }
];

export const DASHBOARD_STATS = {
  totalStudents: 235,
  totalScreenings: 524,
  highRiskCount: 36,
  avgScore: 0.48
};

export const QUESTIONNAIRE_CONFIG = {
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

// Utility to find student by enrollment ID
export function findStudentByEnrollment(enrollmentId) {
  return MOCK_STUDENTS.find(
    s => s.enrollment_id.toLowerCase() === enrollmentId.toLowerCase()
  );
}

// Simulate prediction (returns after a short delay)
export function simulatePrediction(responseData) {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Simple mock scoring based on input values
      const weights = {
        academic_pressure: 0.18,
        financial_stress: 0.15,
        study_satisfaction: -0.12,
        sleep_duration: -0.10,
        work_study_hours: 0.14,
        suicidal_thoughts: 0.25,
        family_history: 0.10
      };
      
      let score = 0.15; // base
      for (const [key, weight] of Object.entries(weights)) {
        const val = responseData[key] ?? 0;
        const normalizedVal = key === 'sleep_duration' ? (4 - val) / 4 : val / 5;
        score += normalizedVal * weight;
      }
      
      score = Math.max(0.05, Math.min(0.98, score));
      
      let riskLevel, recommendation;
      if (score >= 0.7) {
        riskLevel = "High";
        recommendation = RISK_RECOMMENDATIONS.High.message;
      } else if (score >= 0.4) {
        riskLevel = "Moderate";
        recommendation = RISK_RECOMMENDATIONS.Moderate.message;
      } else {
        riskLevel = "Low";
        recommendation = RISK_RECOMMENDATIONS.Low.message;
      }
      
      resolve({
        probability_score: Math.round(score * 100) / 100,
        risk_level: riskLevel,
        recommendation
      });
    }, 2000);
  });
}
