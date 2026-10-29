---
oem: ExampleCo
model: CP-200
title: CP-200 End-Suction Centrifugal Pump Service Manual
license: Apache-2.0. Original synthetic text written for this repository.
---
<!-- page: 1 -->

# CP-200 End-Suction Centrifugal Pump Service Manual

This is a synthetic manual written for the service-manual-rag sample corpus. The manufacturer, equipment, part numbers and values are fictional. Do not use it to service real equipment.

## Safety

### Lockout and tagout procedure

A stopped pump can restart from a level switch or a remote controller. Complete these steps before you remove a guard, open the casing or open the terminal box.

1. Stop the pump at the local control station.
2. Open the motor supply isolator. Fit a personal padlock and a tag with your name, the date and the reason for isolation.
3. Close the suction and discharge isolation valves and tag both handwheels.
4. Press START at the local control station to prove that the motor does not run. Return the selector to OFF.
5. Verify absence of voltage at the motor terminals with a tested voltage indicator.
6. Open the casing drain plug (EC-200-114). Wait until flow from the drain stops before you loosen any other fastener.

Each person who works on the pump fits a personal lock. Allow the casing to cool below 40 °C before work starts. At the maximum liquid temperature of 90 °C the casing surface exceeds 80 °C.

### Lifting the pump unit

The complete unit weighs 98 kg: the motor weighs 62 kg and the pump end weighs 36 kg. The motor eyebolt is rated for the motor alone. Do not lift the complete unit by the eyebolt, because the unit tilts toward the pump end and overloads the eyebolt.

1. Use slings and a hoist rated for at least 150 kg.
2. Place one sling around the motor frame next to the adapter.
3. Place the second sling around the pump casing next to the discharge flange.
4. Lift until the unit hangs level, then lower it onto the foundation bolts.

<!-- page: 2 -->

## Specifications

### Performance and rating data

| Parameter | Value |
|---|---|
| Pump type | End-suction, single-stage, close-coupled |
| Rated flow and head | 60 m³/h at 32 m |
| Shutoff head | 38 m |
| Rated speed | 2,900 rpm |
| Motor | 7.5 kW, 2-pole, 400 V, 50 Hz, TEFC |
| Motor rated current | 14.0 A at 400 V |
| NPSH required at rated flow | 3.5 m |
| Maximum working pressure | 10 bar |
| Liquid temperature | 0 to 90 °C |
| Suction and discharge flanges | DN 80 and DN 65, PN 16 |
| Standard impeller diameter | 180 mm |

### Materials of construction

| Component | Material | Part number |
|---|---|---|
| Casing | Grey cast iron | EC-200-101 |
| Impeller | Cast stainless steel, closed, 6 vanes | EC-200-130 |
| Wear ring | Bronze | EC-200-140 |
| Mechanical seal | Carbon against silicon carbide, EPDM elastomers | EC-200-210 |
| Casing gasket | Aramid fiber sheet, 1.0 mm | EC-200-118 |

The mechanical seal is a single rubber bellows seal for a 28 mm shaft. Mineral oil and grease cause EPDM to swell. Lubricate the elastomers during assembly with soapy water only.

### Fastener torque values

| Fastener | Size | Torque |
|---|---|---|
| Impeller nut | M16 | 55 N·m |
| Casing to adapter bolts (8 off) | M12 | 70 N·m |
| Adapter to motor flange bolts | M10 | 45 N·m |
| Pipe flange bolts | M16 | 80 N·m |
| Foundation bolts | M16 | 110 N·m |
| Drain plug and vent plug | G 3/8 | 25 N·m |

Tighten the casing bolts in a cross pattern in two stages, first to 35 N·m and then to 70 N·m. The impeller nut (EC-200-131) has a nylon locking insert. Replace it each time it is removed.

## Installation

### Suction and discharge piping

Support all piping independently so that the pump flanges carry no pipe weight. Each pipe flange must be parallel to the pump flange within 0.5 mm before bolting.

<!-- page: 3 -->

- Use a suction pipe of at least DN 100 with an eccentric reducer, flat side on top, at the pump flange.
- Provide at least 500 mm of straight suction pipe upstream of the reducer.
- Keep the NPSH available at least 1.0 m above the NPSH required. At rated flow the minimum NPSH available is 4.5 m.
- Fit a check valve and an isolation valve in the discharge line, with the check valve nearest to the pump.
- Fit a temporary 2 mm mesh strainer in the suction line for the first 24 operating hours.

### Electrical connection and rotation check

On a 400 V, 50 Hz supply connect the motor in delta for direct on line starting. Use 25 A time-delay fuses and set the motor protection relay to 14.0 A.

The correct rotation is clockwise viewed from the motor fan end, as shown by the arrow on the adapter.

1. Prime the pump first. Do not check rotation with an empty casing.
2. Start the motor and stop it again within 1 second.
3. Watch the fan through the fan cover grille as it slows down.
4. If the direction is wrong, isolate the supply and swap two phase conductors at the motor terminals.

## Operation

### Priming and first start

The CP-200 is not self-priming. The seal faces are destroyed within 30 seconds of dry running.

1. Open the suction valve fully. Close the discharge valve, then open it a quarter turn.
2. Remove the vent plug (EC-200-115) on top of the casing.
3. Fill the casing until water without air bubbles comes out of the vent.
4. Refit the vent plug and tighten it to 25 N·m.
5. Start the motor. Within 2 minutes, open the discharge valve slowly until the gauge shows the duty pressure.
6. A new seal can leak a few drops during the first 30 minutes while the faces bed in. Leakage after 30 minutes indicates a seal fault.
7. Record suction pressure, discharge pressure and motor current in the commissioning record.

<!-- page: 4 -->

### Operating limits

| Parameter | Limit |
|---|---|
| Minimum continuous flow | 10 m³/h |
| Maximum flow | 75 m³/h |
| Maximum time against a closed discharge valve | 2 minutes |
| Maximum suction pressure | 6 bar |
| Maximum starts per hour | 20, evenly spaced |
| Frequency range on a variable speed drive | 25 to 50 Hz |

Below the minimum flow the water in the casing heats and can flash to steam. Above the maximum flow the pump cavitates. Use the discharge valve to correct the flow. Do not throttle the suction valve.

### Shutdown and frost protection

1. Close the discharge valve slowly to limit pressure surge.
2. Stop the motor.
3. Close the suction valve if the pump will stay out of service.

If the ambient temperature can fall below 2 °C during standstill, open the drain plug and the vent plug and leave both open. Tag the isolator to state that the casing is drained. Before the next start, refit both plugs with new sealing washers at 25 N·m and prime the pump.

## Maintenance

### Maintenance schedule

| Interval | Task |
|---|---|
| Daily | Check the seal for leakage and listen for abnormal noise |
| 500 operating hours | Compare discharge pressure and motor current with the commissioning record |
| 4,000 operating hours | Regrease the motor drive end bearing |
| 8,000 operating hours or 2 years | Measure the wear ring clearance and replace the casing gasket |
| 16,000 operating hours | Replace the mechanical seal |
| 20,000 operating hours | Replace both motor bearings |

### Bearing lubrication interval

The motor carries the impeller on its shaft extension. The drive end bearing (6308 C3) is regreasable. The non-drive end bearing (6208-2Z C3) is sealed for life.

Regrease the drive end bearing every 4,000 operating hours with 10 g of lithium complex NLGI 2 grease. Reduce the interval to 2,000 operating hours when the liquid temperature exceeds 70 °C or the ambient temperature exceeds 35 °C. Do not mix greases with different thickener types.

<!-- page: 5 -->

1. Clean the grease nipple on the drive end bearing cover.
2. Remove the grease outlet plug at the bottom of the bearing cover.
3. With the pump running, inject 10 g of grease, which is 8 strokes of a gun that delivers 1.25 g per stroke.
4. Run the pump for 1 hour with the outlet plug removed, then refit it.

The bearing housing temperature can rise by 10 to 15 °C for a few hours after regreasing. It must not exceed 80 °C.

### Mechanical seal replacement

Seal kit EC-200-210 contains the rotating unit, the stationary seat and a casing gasket. Order a new impeller nut (EC-200-131) with it. The back pull-out design leaves the casing connected to the piping.

1. Isolate and drain the pump as described in the lockout and tagout procedure.
2. Remove the 8 M12 casing bolts. Pull the motor, adapter and impeller out of the casing as one assembly.
3. Hold the impeller with a strap wrench and remove the impeller nut (right-hand thread).
4. Pull off the impeller and remove the key.
5. Slide the rotating unit off the shaft. Press the stationary seat out of the adapter bore.
6. Clean the bore and shaft. Shaft runout at the seal position must not exceed 0.05 mm.
7. Wet the new seat cup with soapy water and press it in by hand, lapped face toward the impeller. Do not touch the lapped face.
8. Wet the shaft and bellows with soapy water and push the rotating unit on until it touches the seat.
9. Refit the key and impeller. The impeller hub sets the seal working length of 34 mm.
10. Fit the new impeller nut at 55 N·m.
11. Fit the new casing gasket (EC-200-118) and refit the assembly. Tighten the casing bolts to 70 N·m in two stages.
12. Turn the fan by hand to confirm that the shaft turns without rubbing. Prime the pump before the start.

<!-- page: 6 -->

### Impeller and wear ring clearance

Measure the clearance every 8,000 operating hours, or when the discharge pressure has fallen by more than 10% at the same flow. A worn clearance lets water recirculate from discharge to suction.

1. Remove the impeller as described in steps 1 to 4 of the mechanical seal replacement.
2. Measure the impeller neck diameter at 3 positions 120° apart. The nominal diameter is 110.00 mm.
3. Measure the wear ring bore at the same 3 positions.
4. The diametral clearance is the smallest bore reading minus the largest neck reading.

| Condition | Diametral clearance |
|---|---|
| New pump | 0.30 to 0.40 mm |
| Replace wear ring | 0.80 mm or more |

The wear ring (EC-200-140) is pressed into the casing with an interference of 0.05 to 0.10 mm. Replace the impeller as well if the neck has worn below 109.70 mm or the vane inlet edges show cavitation pitting deeper than 2 mm.

## Troubleshooting

### Low discharge pressure

Compare the discharge pressure with the commissioning record at the same flow.

| Possible cause | Check | Corrective action |
|---|---|---|
| Reverse rotation after electrical work | Pump delivers about 50% of rated head. Fan turns counterclockwise viewed from the fan end. | Isolate and swap two phase conductors at the motor terminals |
| Air in the casing or suction line | Air escapes at the vent plug | Prime again and check suction joints for leaks |
| Suction valve partly closed or strainer blocked | Suction gauge below the commissioning value | Open the valve fully and clean or remove the strainer |
| Worn wear ring | Diametral clearance 0.80 mm or more | Replace wear ring EC-200-140 |

### Excessive noise and vibration

Measure vibration as RMS velocity on the drive end bearing housing. A new installation must run below 2.8 mm/s. Investigate any reading above 4.5 mm/s.

<!-- page: 7 -->

| Symptom | Possible cause | Corrective action |
|---|---|---|
| Crackling noise in the casing | Cavitation from low NPSH available | Open the suction valve fully, clean the strainer and reduce flow |
| Rumbling and fluctuating pressure at low flow | Flow below the minimum continuous flow | Open the discharge valve or fit a bypass line |
| High pitched noise, bearing housing above 80 °C | Worn or underlubricated drive end bearing | Regrease. Replace the bearing if the noise remains. |
| Vibration varies with bolt tightness | Loose foundation bolts | Tighten to 110 N·m |

### Seal leakage

| Symptom | Possible cause | Corrective action |
|---|---|---|
| Steady drip after the 30 minute bedding period | Faces damaged by dry running | Replace seal kit EC-200-210 |
| Leakage directly after a seal change | Seat not square in the bore, or elastomer swollen by oil | Refit with soapy water as the only lubricant |
| Leakage after a frost period | Cracked stationary seat | Replace the seal kit and drain the casing before frost |

### Motor protection relay trips

| Possible cause | Check | Corrective action |
|---|---|---|
| Flow above 75 m³/h because system head is lower than design | Motor current above 14.0 A | Throttle the discharge valve until the current is 14.0 A or less |
| Phase loss | One phase current reads zero | Replace the open fuse and find the cause |
| Supply voltage out of tolerance | Voltage outside 380 to 420 V | Correct the supply |
