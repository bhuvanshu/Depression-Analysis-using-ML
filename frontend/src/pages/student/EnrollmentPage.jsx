import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, ArrowRight, AlertCircle, Brain } from 'lucide-react';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import { findStudentByEnrollment } from '../../data/mockData';
import './EnrollmentPage.css';

export default function EnrollmentPage() {
  const navigate = useNavigate();
  const [enrollmentId, setEnrollmentId] = useState('');
  const [student, setStudent] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleVerify = (e) => {
    e.preventDefault();
    if (!enrollmentId.trim()) {
      setError('Please enter your enrollment ID');
      return;
    }

    setLoading(true);
    setError('');
    setStudent(null);

    // Simulate API lookup
    setTimeout(() => {
      const found = findStudentByEnrollment(enrollmentId.trim());
      if (found) {
        setStudent(found);
      } else {
        setError('No student found with this enrollment ID. Please check and try again.');
      }
      setLoading(false);
    }, 800);
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
          <h1 className="enrollment-title">Mental Health Screening</h1>
          <p className="enrollment-subtitle">
            Confidential depression risk assessment for students
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
                    <div className="student-preview-id">{student.enrollment_id}</div>
                  </div>
                </div>

                <div className="student-preview-grid">
                  <div className="student-preview-item">
                    <span className="student-preview-label">Department</span>
                    <span className="student-preview-value">{student.department}</span>
                  </div>
                  <div className="student-preview-item">
                    <span className="student-preview-label">Degree</span>
                    <span className="student-preview-value">{student.degree_group}</span>
                  </div>
                  <div className="student-preview-item">
                    <span className="student-preview-label">Age</span>
                    <span className="student-preview-value">{student.age} years</span>
                  </div>
                  <div className="student-preview-item">
                    <span className="student-preview-label">CGPA</span>
                    <span className="student-preview-value">{student.cgpa}</span>
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

        <p className="enrollment-footer">
          Are you an admin? <a href="/admin/login">Login here</a>
        </p>
      </div>
    </div>
  );
}
