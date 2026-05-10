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
  const inputId = id || `input-${label?.replace(/\s+/g, '-').toLowerCase()}`;

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
          type={type}
          className={`input-field ${Icon ? 'has-icon' : ''} ${readOnly ? 'read-only' : ''}`}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          disabled={disabled}
          readOnly={readOnly}
          required={required}
          {...props}
        />
      </div>
      {error && <span className="input-error">{error}</span>}
      {hint && !error && <span className="input-hint">{hint}</span>}
    </div>
  );
}
