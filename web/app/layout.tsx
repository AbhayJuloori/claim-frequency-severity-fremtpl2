import "./globals.css";

export const metadata = {
  title: "Claim Pricing Explorer",
  description: "Claim frequency and severity models on freMTPL2: GLM vs LightGBM, segment comparison, checked notes.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
