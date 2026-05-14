import {
  User, Hash, Building2, GraduationCap, Calendar,
  ShieldAlert, TrendingUp, Activity, ClipboardList,
  AlertTriangle, CheckCircle2, AlertCircle
} from 'lucide-react';
import Modal from '../../components/common/Modal';
import RiskBadge from '../../components/common/RiskBadge';
import './StudentDetailModal.css';

export default function StudentDetailModal({ isOpen, student, onClose }) {
  if (!student) return null;

  const score = ((student.probabilityScore || 0) * 100).toFixed(1);
  const riskLevel = student.riskLevel || 'Low';
  const riskClass = riskLevel.toLowerCase();

  // Generate recommendation based on risk level
  const getRecommendation = (level) => {
    switch (level) {
      case 'High':
        return {
          text: 'Immediate professional counseling referral recommended. Schedule follow-up within 48 hours.',
          icon: AlertTriangle,
          color: 'danger'
        };
      case 'Moderate':
        return {
          text: 'Recommend wellness check-in and monitoring. Consider peer support group enrollment.',
          icon: AlertCircle,
          color: 'warning'
        };
      default:
        return {
          text: 'No immediate intervention needed. Continue regular screening schedule.',
          icon: CheckCircle2,
          color: 'success'
        };
    }
  };

  const recommendation = getRecommendation(riskLevel);
  const RecIcon = recommendation.icon;

  // Map questionnaire fields from the student data (if available)
  const questionnaireFields = [
    { label: 'Academic Pressure', key: 'academicPressure', max: 5 },
    { label: 'Study Satisfaction', key: 'studySatisfaction', max: 5 },
    { label: 'Sleep Duration', key: 'sleepDuration', suffix: ' hrs' },
    { label: 'Dietary Habits', key: 'dietaryHabits', max: 5 },
    { label: 'Suicidal Thoughts', key: 'suicidalThoughts', type: 'yesno' },
    { label: 'Study Hours', key: 'studyHours', suffix: ' hrs/day' },
    { label: 'Financial Stress', key: 'financialStress', max: 5 },
    { label: 'Family History', key: 'familyHistory', type: 'yesno' },
  ];

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Student Details" maxWidth="640px">
      <div className="student-detail">
        {/* ── Profile Header ── */}
        <div className={`detail-profile detail-profile-${riskClass}`}>
          <div className="detail-avatar">
            {student.studentName ? student.studentName.split(' ').map(n => n[0]).join('').toUpperCase() : '?'}
          </div>
          <div className="detail-profile-info">
            <h3>{student.studentName}</h3>
            <div className="detail-profile-meta">
              <span><Hash size={13} /> {student.enrollmentId}</span>
              <span><Building2 size={13} /> {student.department}</span>
            </div>
          </div>
          <RiskBadge level={riskLevel} size="lg" />
        </div>

        {/* ── Risk Assessment ── */}
        <div className="detail-section">
          <h4 className="detail-section-title">
            <ShieldAlert size={16} /> Risk Assessment
          </h4>
          <div className="risk-assessment-grid">
            <div className="risk-metric">
              <div className="risk-metric-label">Probability Score</div>
              <div className={`risk-metric-value score-${riskClass}`}>{score}%</div>
              <div className="risk-meter">
                <div
                  className={`risk-meter-fill risk-meter-${riskClass}`}
                  style={{ width: `${score}%` }}
                />
              </div>
            </div>
            <div className="risk-metric">
              <div className="risk-metric-label">Risk Category</div>
              <div className={`risk-metric-value score-${riskClass}`}>{riskLevel}</div>
            </div>
            {student.screeningDate && (
              <div className="risk-metric">
                <div className="risk-metric-label">Screening Date</div>
                <div className="risk-metric-value">
                  {new Date(student.screeningDate).toLocaleDateString('en-IN', {
                    day: '2-digit', month: 'short', year: 'numeric'
                  })}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Recommendation ── */}
        <div className={`detail-recommendation detail-recommendation-${recommendation.color}`}>
          <RecIcon size={18} />
          <div>
            <strong>Recommendation</strong>
            <p>{recommendation.text}</p>
          </div>
        </div>

        {/* ── Questionnaire Responses ── */}
        <div className="detail-section">
          <h4 className="detail-section-title">
            <ClipboardList size={16} /> Screening Responses
          </h4>
          <div className="responses-grid">
            {questionnaireFields.map(field => {
              const value = student[field.key];
              if (value === undefined || value === null) return null;

              let displayValue;
              if (field.type === 'yesno') {
                displayValue = value === 'Yes' || value === true || value === 1 ? 'Yes' : 'No';
              } else if (field.suffix) {
                displayValue = `${value}${field.suffix}`;
              } else if (field.max) {
                displayValue = `${value} / ${field.max}`;
              } else {
                displayValue = String(value);
              }

              return (
                <div key={field.key} className="response-item">
                  <span className="response-label">{field.label}</span>
                  <span className={`response-value ${field.type === 'yesno' && (value === 'Yes' || value === true || value === 1) ? 'response-flag' : ''}`}>
                    {displayValue}
                  </span>
                </div>
              );
            })}
            {questionnaireFields.every(f => student[f.key] === undefined || student[f.key] === null) && (
              <div className="responses-empty">
                <Activity size={20} />
                <p>Detailed questionnaire responses are not available for this screening.</p>
              </div>
            )}
          </div>
        </div>

        {/* ── Student Info ── */}
        <div className="detail-section">
          <h4 className="detail-section-title">
            <User size={16} /> Student Information
          </h4>
          <div className="info-grid">
            {student.age && (
              <div className="info-item">
                <span className="info-label">Age</span>
                <span className="info-value">{student.age}</span>
              </div>
            )}
            {student.gender && (
              <div className="info-item">
                <span className="info-label">Gender</span>
                <span className="info-value">{student.gender}</span>
              </div>
            )}
            {student.degreeGroup && (
              <div className="info-item">
                <span className="info-label">Degree</span>
                <span className="info-value">{student.degreeGroup}</span>
              </div>
            )}
            {student.department && (
              <div className="info-item">
                <span className="info-label">Department</span>
                <span className="info-value">{student.department}</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </Modal>
  );
}
