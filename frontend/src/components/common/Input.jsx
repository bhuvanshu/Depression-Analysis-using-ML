import { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import './Input.css';

export default function Input({
  label,
  type = 'text',
  value,
  onChange,
  placeholder,
  icon: Icon,
  required = false,
  disabled = false,
  readOnly = false,
  error,
  hint,
  className = '',
  id,
  ...props
}) {
  const [showPassword, setShowPassword] = useState(false);
  const inputId = id || `input-${label?.replace(/\s+/g, '-').toLowerCase()}`;

  const isPassword = type === 'password';
  const inputType = isPassword ? (showPassword ? 'text' : 'password') : type;

  if (type === 'select') {
    return (
      <div className={`input-group ${className}`}>
        {label && (
          <label className="input-label" htmlFor={inputId}>
            {label} {required && <span className="required">*</span>}
          </label>
        )}
        <div className="input-wrapper">
          {Icon && <Icon size={18} className="input-icon" />}
          <select
            id={inputId}
            className={`input-field select-field ${Icon ? 'has-icon' : ''}`}
            value={value}
            onChange={onChange}
            disabled={disabled}
            {...props}
          >
            {props.children}
          </select>
        </div>
        {error && <span className="input-error">{error}</span>}
        {hint && !error && <span className="input-hint">{hint}</span>}
      </div>
    );
  }

  if (type === 'textarea') {
    return (
      <div className={`input-group ${className}`}>
        {label && (
          <label className="input-label" htmlFor={inputId}>
            {label} {required && <span className="required">*</span>}
          </label>
        )}
        <textarea
          id={inputId}
          className={`input-field textarea-field ${readOnly ? 'read-only' : ''}`}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          disabled={disabled}
          readOnly={readOnly}
          {...props}
        />
        {error && <span className="input-error">{error}</span>}
        {hint && !error && <span className="input-hint">{hint}</span>}
      </div>
    );
  }

  return (
    <div className={`input-group ${className}`}>
      {label && (
        <label className="input-label" htmlFor={inputId}>
          {label} {required && <span className="required">*</span>}
        </label>
      )}
      <div className="input-wrapper">
        {Icon && <Icon size={18} className="input-icon" />}
        <input
          id={inputId}
          type={inputType}
          className={`input-field ${Icon ? 'has-icon' : ''} ${readOnly ? 'read-only' : ''} ${isPassword ? 'has-password-toggle' : ''}`}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          disabled={disabled}
          readOnly={readOnly}
          required={required}
          {...props}
        />
        {isPassword && (
          <button
            type="button"
            className="password-toggle-btn"
            onClick={() => setShowPassword(!showPassword)}
            tabIndex={-1}
            aria-label={showPassword ? "Hide password" : "Show password"}
          >
            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        )}
      </div>
      {error && <span className="input-error">{error}</span>}
      {hint && !error && <span className="input-hint">{hint}</span>}
    </div>
  );
}
