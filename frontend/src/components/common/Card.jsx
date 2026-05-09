import './Card.css';

export default function Card({ 
  children, 
  hoverable = false, 
  elevated = false,
  accent,
  className = '',
  ...props 
}) {
  const classes = [
    'card',
    hoverable && 'card-hoverable',
    elevated && 'card-elevated',
    accent && `card-accent-${accent}`,
    className
  ].filter(Boolean).join(' ');

  return (
    <div className={classes} {...props}>
      {children}
    </div>
  );
}
