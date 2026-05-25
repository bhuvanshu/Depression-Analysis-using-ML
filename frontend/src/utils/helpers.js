export const getInitials = (name) =>
  name ? name.split(' ').map(n => n[0]).join('').toUpperCase() : '?';
