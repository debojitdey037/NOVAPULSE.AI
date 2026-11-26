# Power Subsystem & Battery Management Architecture

## 1. Split Cell Topology & Ergonomics
To ensure ideal 50/50 lateral weight balance and zero ear fatigue over extended mastering sessions, the power storage is bifurcated:
- **Capacity**: Dual 600 mAh Lithium-Polymer cells (1200 mAh combined total, 4.44 Wh).
- One cell is housed in the left acoustic isolation compartment; the matching balance weight is mirrored on the right chamber.
- Target Exploded Frame: 142

## 2. Playback Endurance Metrics
- Standard SBC/AAC mode (ANC Disabled): Up to 68 hours continuous playback at 75 dB SPL.
- Audiophile LDAC 990kbps + Hybrid Neural ANC Active: 48 hours continuous playback.
- Standby Drain: Smart ultra-low power sleep states (<15 microamps) maintain battery charge for up to 180 days on idle.

## 3. WarpCharge 3.0 & Thermal Protections
- Connector: USB Type-C with USB Power Delivery 3.0 PPS support.
- Rapid Charging: 15 minutes of charging delivers 8.5 hours of playback. Full 0 to 100% replenishment is achieved in 58 minutes.
- Safety: Integrated thermal sensor monitoring with over-voltage, reverse-polarity, and pouch swelling prevention ICs.
