import './Loader.css';

export default function Loader({ size = 'md', text, fullscreen = false }) {
  const content = (
    <div className={`loader-overlay ${fullscreen ? 'loader-fullscreen' : ''}`}>
      <div className={`loader-spinner ${size}`} />
      {text && <p className="loader-text">{text}</p>}
    </div>
  );

  return content;
}

export function Skeleton({ type = 'text', width, height, className = '' }) {
  const style = {};
  if (width) style.width = width;
  if (height) style.height = height;

  return <div className={`skeleton skeleton-${type} ${className}`} style={style} />;
}
