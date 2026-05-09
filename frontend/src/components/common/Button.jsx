import './Button.css';

export default function Button({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  fullWidth = false, 
  loading = false, 
  disabled = false, 
  icon: Icon,
  onClick, 
  type = 'button',
  className = '',
  ...props 
}) {
  const classes = [
    'btn',
    `btn-${variant}`,
    size !== 'md' && `btn-${size}`,
    fullWidth && 'btn-full',
    className
  ].filter(Boolean).join(' ');

  return (
    <button
      type={type}
      className={classes}
      onClick={onClick}
      disabled={disabled || loading}
      {...props}
    >
      {loading && <span className="btn-spinner" />}
      {!loading && Icon && <Icon size={size === 'sm' ? 14 : 18} />}
      {children}
    </button>
  );
}
