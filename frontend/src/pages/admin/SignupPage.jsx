import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Mail, Lock, UserPlus, Brain, Building2, User, BarChart3, Shield, Users } from 'lucide-react';
import Button from '../../components/common/Button';
import Input from '../../components/common/Input';
import { adminSignup } from '../../services/api';
import './AuthPage.css';

export default function SignupPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    collegeName: '',
    adminName: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (field) => (e) => {
    setForm(prev => ({ ...prev, [field]: e.target.value }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.collegeName || !form.adminName || !form.email || !form.password || !form.confirmPassword) {
      setError('Please fill in all fields');
      return;
    }
    if (form.password !== form.confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    if (form.password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);
    try {
      await adminSignup({
        collegeName: form.collegeName,
        adminName: form.adminName,
        adminEmail: form.email,
        password: form.password
      });
      
      // Store mock auth state for frontend routing (we haven't set up real JWTs yet)
      localStorage.setItem('admin_auth', JSON.stringify({
        email: form.email,
        name: form.adminName,
        college: form.collegeName
      }));
      navigate('/admin/dashboard');
    } catch (err) {
      setError(err.message || 'Signup failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      {/* Branding Side */}
      <div className="auth-branding">
        <div className="auth-branding-content">
          <div className="auth-branding-logo">
            <Brain size={28} color="white" />
          </div>
          <h1>Join the Platform</h1>
          <p>
            Register your institution to start monitoring and supporting
            student mental health with our AI-powered screening system.
          </p>
          <div className="auth-features">
            <div className="auth-feature">
              <div className="auth-feature-icon"><BarChart3 size={18} /></div>
              Department-wise analytics & trend reports
            </div>
            <div className="auth-feature">
              <div className="auth-feature-icon"><Shield size={18} /></div>
              Confidential, HIPAA-compliant data handling
            </div>
            <div className="auth-feature">
              <div className="auth-feature-icon"><Users size={18} /></div>
              Bulk student upload & management
            </div>
          </div>
        </div>
      </div>

      {/* Form Side */}
      <div className="auth-form-side">
        <div className="auth-form-container">
          <div className="auth-form-header">
            <h2>Create Account</h2>
            <p>Register your institution to get started</p>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            <Input
              label="College / Institution Name"
              icon={Building2}
              placeholder="National Institute of Technology"
              value={form.collegeName}
              onChange={handleChange('collegeName')}
              required
            />
            <Input
              label="Admin Name"
              icon={User}
              placeholder="Dr. Sharma"
              value={form.adminName}
              onChange={handleChange('adminName')}
              required
            />
            <Input
              label="Email Address"
              type="email"
              icon={Mail}
              placeholder="admin@college.edu"
              value={form.email}
              onChange={handleChange('email')}
              required
            />
            <div className="auth-form-row">
              <Input
                label="Password"
                type="password"
                icon={Lock}
                placeholder="Min 6 characters"
                value={form.password}
                onChange={handleChange('password')}
                required
              />
              <Input
                label="Confirm Password"
                type="password"
                icon={Lock}
                placeholder="Re-enter password"
                value={form.confirmPassword}
                onChange={handleChange('confirmPassword')}
                required
              />
            </div>

            {error && <div className="auth-error">{error}</div>}

            <Button type="submit" variant="primary" size="lg" fullWidth loading={loading} icon={UserPlus}>
              Create Account
            </Button>
          </form>

          <div className="auth-form-footer">
            Already have an account? <Link to="/admin/login">Sign in</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
