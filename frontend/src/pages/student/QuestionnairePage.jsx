import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, ArrowRight, Send, Shield, Heart, BookOpen, Moon, Clock, Users, CheckCircle } from 'lucide-react';
import Button from '../../components/common/Button';
import Card from '../../components/common/Card';
import Input from '../../components/common/Input';
import { QUESTIONNAIRE_CONFIG } from '../../data/mockData';
import { submitScreening } from '../../services/api';
import './QuestionnairePage.css';

export default function QuestionnairePage() {
  const location = useLocation();
  const navigate = useNavigate();
  const student = location.state?.student;

  const [formData, setFormData] = useState({
    academic_pressure: 2,
    financial_stress: 2,
    study_satisfaction: 3,
    sleep_duration: 3,
    work_study_hours: 2,
    suicidal_thoughts: 0,
    family_history: 0,
    other_factors: ''
  });

  const [submitting, setSubmitting] = useState(false);

  // Redirect if no student data
  if (!student) {
    return (
      <div className="questionnaire-page">
        <div className="bg-pattern" />
        <div className="questionnaire-container" style={{ textAlign: 'center', paddingTop: '20vh' }}>
          <h2>No student data found</h2>
          <p style={{ color: 'var(--text-secondary)', margin: 'var(--space-4) 0' }}>
            Please verify your enrollment first.
          </p>
          <Button variant="primary" onClick={() => navigate('/')}>
            Go to Enrollment
          </Button>
        </div>
      </div>
    );
  }

  const handleSliderChange = (key, value) => {
    setFormData(prev => ({ ...prev, [key]: parseInt(value) }));
  };

  const handleToggle = (key, value) => {
    setFormData(prev => ({ ...prev, [key]: value }));
  };

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      const screeningPayload = {
        enrollmentId: student.enrollmentId,
        age: student.age,
        gender: student.gender,
        degree: student.degreeGroup,
        // Optional/mocked values for now if not present in formData:
        cgpa: 8.0, 
        academic_pressure: formData.academic_pressure,
        financial_stress: formData.financial_stress,
        study_satisfaction: formData.study_satisfaction,
        work_study_hours: formData.work_study_hours,
        suicidal_thoughts: formData.suicidal_thoughts === 1,
        family_history: formData.family_history === 1,
        sleep_duration: formData.sleep_duration.toString()
      };

      const result = await submitScreening(screeningPayload);
      
      navigate('/result', { state: { student, formData, result } });
    } catch (err) {
      console.error(err);
      alert('Failed to submit screening. ' + (err.message || ''));
    } finally {
      setSubmitting(false);
    }
  };

  const getInitials = (name) => name.split(' ').map(n => n[0]).join('');

  const sectionIcons = {
    academic_pressure: BookOpen,
    financial_stress: Clock,
    study_satisfaction: Heart,
    sleep_duration: Moon,
    work_study_hours: Clock,
    suicidal_thoughts: Shield,
    family_history: Users
  };

  const renderSlider = (key, config) => {
    const value = formData[key];
    const Icon = sectionIcons[key];
    return (
      <div className="slider-group" key={key}>
        <div className="slider-header">
          <span className="slider-label">
            {Icon && <Icon size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: 'middle', opacity: 0.6 }} />}
            {config.label}
          </span>
          <span className="slider-value">{config.labels[value]}</span>
        </div>
        <input
          type="range"
          className="slider-input"
          min={config.min}
          max={config.max}
          value={value}
          onChange={(e) => handleSliderChange(key, e.target.value)}
        />
        <div className="slider-labels">
          <span>{config.labels[config.min]}</span>
          <span>{config.labels[config.max]}</span>
        </div>
      </div>
    );
  };

  const renderToggle = (key, config) => {
    const value = formData[key];
    const isDanger = key === 'suicidal_thoughts';
    return (
      <div className="toggle-group" key={key}>
        <div className="toggle-label">{config.label}</div>
        {config.description && <div className="toggle-description">{config.description}</div>}
        <div className="toggle-options">
          {config.options.map(opt => (
            <button
              key={opt.value}
              type="button"
              className={`toggle-option ${value === opt.value ? (isDanger && opt.value === 1 ? 'selected-danger' : 'selected') : ''}`}
              onClick={() => handleToggle(key, opt.value)}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>
    );
  };

  const renderSelect = (key, config) => (
    <div key={key}>
      <Input
        type="select"
        label={config.label}
        value={formData[key]}
        onChange={(e) => handleSliderChange(key, e.target.value)}
        icon={Moon}
      >
        {config.options.map(opt => (
          <option key={opt.value} value={opt.value}>{opt.label}</option>
        ))}
      </Input>
    </div>
  );

  return (
    <div className="questionnaire-page">
      <div className="bg-pattern" />

      <div className="questionnaire-container">
        {/* Progress */}
        <div className="progress-bar-container">
          <div className="progress-steps">
            <div className="progress-step done">
              <span className="progress-step-dot"><CheckCircle size={14} /></span>
              <span>Verify</span>
            </div>
            <span className="progress-step-line done" />
            <div className="progress-step active">
              <span className="progress-step-dot">2</span>
              <span>Questionnaire</span>
            </div>
            <span className="progress-step-line" />
            <div className="progress-step">
              <span className="progress-step-dot">3</span>
              <span>Result</span>
            </div>
          </div>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: '50%' }} />
          </div>
        </div>

        {/* Student Banner */}
        <div className="student-info-banner">
          <div className="student-avatar">{getInitials(student.name)}</div>
          <div className="student-info-details">
            <div className="student-info-name">{student.name}</div>
            <div className="student-info-meta">
              {student.enrollmentId} · {student.department}
            </div>
          </div>
        </div>

        {/* Header */}
        <div className="questionnaire-header">
          <h1>Screening Questionnaire</h1>
          <div className="sensitive-notice" style={{ textAlign: 'left', margin: '0 auto', maxWidth: '100%' }}>
            <Shield size={16} />
            <span>
              The following questions are part of a standardized screening process.
              Your responses are kept strictly confidential and are used only to connect you with appropriate support.
            </span>
          </div>
        </div>

        {/* Form */}
        <Card elevated>
          <div className="questionnaire-form">
            {/* Stress & Academic Section */}
            <div className="form-section">
              <div className="form-section-title">
                <BookOpen size={14} /> Academic & Financial
              </div>
              {renderSlider('academic_pressure', QUESTIONNAIRE_CONFIG.academic_pressure)}
              {renderSlider('financial_stress', QUESTIONNAIRE_CONFIG.financial_stress)}
              {renderSlider('study_satisfaction', QUESTIONNAIRE_CONFIG.study_satisfaction)}
            </div>

            {/* Lifestyle Section */}
            <div className="form-section delay-1">
              <div className="form-section-title">
                <Moon size={14} /> Lifestyle
              </div>
              {renderSelect('sleep_duration', QUESTIONNAIRE_CONFIG.sleep_duration)}
              {renderSlider('work_study_hours', QUESTIONNAIRE_CONFIG.work_study_hours)}
            </div>

            {/* Health History Section */}
            <div className="form-section delay-2">
              <div className="form-section-title">
                <Heart size={14} /> Health & History
              </div>

              {renderToggle('suicidal_thoughts', QUESTIONNAIRE_CONFIG.suicidal_thoughts)}
              {renderToggle('family_history', QUESTIONNAIRE_CONFIG.family_history)}
            </div>

            {/* Additional */}
            <div className="form-section delay-3">
              <Input
                type="textarea"
                label="Other Factors (Optional)"
                placeholder="Any other factors you'd like to mention — relationship stress, health issues, etc."
                value={formData.other_factors}
                onChange={(e) => setFormData(prev => ({ ...prev, other_factors: e.target.value }))}
              />
            </div>

            {/* Actions */}
            <div className="questionnaire-actions">
              <Button variant="secondary" icon={ArrowLeft} onClick={() => navigate(-1)}>
                Back
              </Button>
              <Button
                variant="primary"
                icon={submitting ? undefined : Send}
                loading={submitting}
                onClick={handleSubmit}
              >
                {submitting ? 'Analyzing...' : 'Submit Screening'}
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
