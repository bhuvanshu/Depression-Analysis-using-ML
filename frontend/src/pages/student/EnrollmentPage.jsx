import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, ArrowRight, AlertCircle, Brain, Check } from 'lucide-react';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import { verifyStudent } from '../../services/api';
import './EnrollmentPage.css';

export default function EnrollmentPage() {
  const navigate = useNavigate();
  const [enrollmentId, setEnrollmentId] = useState('');
  const [student, setStudent] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleVerify = async (e) => {
    e.preventDefault();
    if (!enrollmentId.trim()) {
      setError('Please enter your enrollment ID');
      return;
    }

    setLoading(true);
    setError('');
    setStudent(null);

    try {
      const foundStudent = await verifyStudent(enrollmentId.trim());
      setStudent(foundStudent);
    } catch (err) {
      setError(err.message || 'No student found with this enrollment ID. Please check and try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleContinue = () => {
    navigate('/questionnaire', { state: { student } });
  };

  const getInitials = (name) => {
    return name.split(' ').map(n => n[0]).join('').toUpperCase();
  };

  return (
    <div className="enrollment-page">
      <div className="bg-pattern" />

      <div className="enrollment-container">
        <div className="enrollment-header">
          <div className="enrollment-logo">
            <Brain size={36} color="white" />
          </div>
          <h1 className="enrollment-title">Mind Care</h1>
          <p className="enrollment-subtitle">
            AI-assisted student mental wellness screening platform
          </p>
        </div>

        <Card elevated>
          <form className="enrollment-form" onSubmit={handleVerify}>
            <div className="input-group">
              <label className="input-label">Enrollment ID</label>
              <div className="input-wrapper">
                <Search size={18} className="input-icon" />
                <input
                  type="text"
                  className="input-field has-icon"
                  placeholder="e.g. BT21CSE001"
                  value={enrollmentId}
                  onChange={(e) => {
                    setEnrollmentId(e.target.value.toUpperCase());
                    setError('');
                    setStudent(null);
                  }}
                  autoFocus
                />
              </div>
            </div>

            {error && (
              <div className="enrollment-error">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            <Button
              type="submit"
              variant="primary"
              size="lg"
              fullWidth
              loading={loading}
              icon={Search}
            >
              Verify Enrollment
            </Button>
          </form>

          {student && (
            <>
              <div className="enrollment-divider">Student Found</div>
              <div className="student-preview">
                <div className="student-preview-header">
                  <div className="student-avatar">
                    {getInitials(student.name)}
                  </div>
                  <div>
                    <div className="student-preview-name">{student.name}</div>
                    <div className="student-preview-id">{student.enrollmentId}</div>
                  </div>
                </div>

                <div className="student-preview-grid">
                  <div className="student-preview-item">
                    <span className="student-preview-label">Department</span>
                    <span className="student-preview-value">{student.department}</span>
                  </div>
                  <div className="student-preview-item">
                    <span className="student-preview-label">Degree</span>
                    <span className="student-preview-value">{student.degreeGroup}</span>
                  </div>
                  <div className="student-preview-item">
                    <span className="student-preview-label">Age</span>
                    <span className="student-preview-value">{student.age} years</span>
                  </div>
                </div>

                <Button
                  variant="primary"
                  size="lg"
                  fullWidth
                  icon={ArrowRight}
                  onClick={handleContinue}
                >
                  Continue to Questionnaire
                </Button>
              </div>
            </>
          )}
        </Card>

        <div className="trust-features">
          <div className="trust-feature">
            <Check size={16} className="trust-icon" />
            <span>Secure</span>
          </div>
          <div className="trust-feature">
            <Check size={16} className="trust-icon" />
            <span>Confidential</span>
          </div>
          <div className="trust-feature">
            <Check size={16} className="trust-icon" />
            <span>AI-assisted</span>
          </div>
        </div>

        <div className="privacy-notice">
          <p>Your responses are confidential and used only for wellness assessment.</p>
        </div>

        <p className="enrollment-footer">
          Are you an admin? <a href="/admin/login">Login here</a>
        </p>
      </div>
    </div>
  );
}
