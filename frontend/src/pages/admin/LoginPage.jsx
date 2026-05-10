import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Mail, Lock, LogIn, Brain, BarChart3, Shield, Users } from 'lucide-react';
import Button from '../../components/common/Button';
import Input from '../../components/common/Input';
import { adminLogin } from '../../services/api';
import './AuthPage.css';

export default function LoginPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({ email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (field) => (e) => {
    setForm(prev => ({ ...prev, [field]: e.target.value }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.email || !form.password) {
      setError('Please fill in all fields');
      return;
    }
    setLoading(true);
    
    try {
      await adminLogin({
        adminEmail: form.email,
        password: form.password
      });

      // Still setting a mock token in local storage so the frontend knows we are logged in
      localStorage.setItem('admin_auth', JSON.stringify({
        email: form.email,
        name: form.email.split('@')[0],
        college: 'Your Institution'
      }));
      navigate('/admin/dashboard');
    } catch (err) {
      setError(err.message || 'Invalid email or password');
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
          <h1>Depression Analysis System</h1>
          <p>
            Monitor and support student mental health with AI-powered screening,
            real-time analytics, and actionable insights for your institution.
          </p>
          <div className="auth-features">
            <div className="auth-feature">
              <div className="auth-feature-icon"><BarChart3 size={18} /></div>
              Real-time risk analytics & department insights
            </div>
            <div className="auth-feature">
              <div className="auth-feature-icon"><Shield size={18} /></div>
              ML-powered depression risk assessment
            </div>
            <div className="auth-feature">
              <div className="auth-feature-icon"><Users size={18} /></div>
              Complete student screening management
            </div>
          </div>
        </div>
      </div>

      {/* Form Side */}
      <div className="auth-form-side">
        <div className="auth-form-container">
          <div className="auth-form-header">
            <h2>Welcome Back</h2>
            <p>Sign in to access your admin dashboard</p>
          </div>

          <form className="auth-form" onSubmit={handleSubmit}>
            <Input
              label="Email Address"
              type="email"
              icon={Mail}
              placeholder="admin@college.edu"
              value={form.email}
              onChange={handleChange('email')}
              required
            />
            <Input
              label="Password"
              type="password"
              icon={Lock}
              placeholder="Enter your password"
              value={form.password}
              onChange={handleChange('password')}
              required
            />

            {error && <div className="auth-error">{error}</div>}

            <Button type="submit" variant="primary" size="lg" fullWidth loading={loading} icon={LogIn}>
              Sign In
            </Button>
          </form>

          <div className="auth-form-footer">
            Don't have an account? <Link to="/admin/signup">Create one</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
