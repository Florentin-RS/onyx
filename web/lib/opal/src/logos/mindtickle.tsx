import type { IconProps } from "@opal/types";
const SvgMindtickle = ({ size, ...props }: IconProps) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 52 52"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    {...props}
  >
    <rect x="2" y="2" width="48" height="48" rx="12" fill="#5B2E91" />
    <path
      d="M12 36V17h5.2l5.3 10.4L27.8 17H33v19h-4.6V24.9l-5.1 9.8h-1.6l-5.1-9.8V36H12Z"
      fill="#FFFFFF"
    />
    <path
      d="M35.5 28.6l3.1 3.1 6.4-7.2"
      stroke="#F26B3A"
      strokeWidth="3"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);
export default SvgMindtickle;
