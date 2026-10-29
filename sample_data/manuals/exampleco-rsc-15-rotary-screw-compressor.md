---
oem: ExampleCo
model: RSC-15
title: RSC-15 Oil-Flooded Rotary Screw Air Compressor Service Manual
license: Apache-2.0. Original synthetic text written for this repository.
---
<!-- page: 1 -->

# RSC-15 Oil-Flooded Rotary Screw Air Compressor Service Manual

This is a synthetic manual written for the service-manual-rag sample corpus. The manufacturer, equipment, part numbers and values are fictional. Do not use it to service real equipment.

## Safety

### Isolation and depressurization procedure

After a stop, the separator tank still holds compressed air and oil at up to 95 °C. Electrical isolation alone does not make the compressor safe to open. Complete every step before you open any part of the air or oil circuit.

1. Press STOP on the controller. The compressor runs unloaded for 30 seconds and stops.
2. Open the main isolator, fit a personal padlock and attach a tag with your name, the date and the reason.
3. Close the outlet ball valve to the air network and tag the handle.
4. Read the separator tank gauge. The blowdown valve vents the tank automatically at stop. Wait until the gauge reads 0 bar.
5. If the gauge still shows pressure 2 minutes after the stop, the blowdown valve is faulty. Open the manual vent valve (EC-15-318) slowly and keep it open until the gauge reads 0 bar.
6. Wait at least 5 minutes after the stop before you open the oil filler plug.
7. Verify absence of voltage at the motor terminals. The cabinet heater has a separate 230 V supply. Isolate it as well.

Allow 30 minutes for the air end, discharge pipe and tank to cool below 50 °C before you touch them. Do not direct compressed air at skin or use it to clean clothing.

### Safety valve function

The safety valve (EC-15-305) on the separator tank opens at 9.5 bar. The tank design pressure is 11 bar. The valve is set and sealed at the factory.

<!-- page: 2 -->

1. Test the valve once a month. With the compressor loaded and the tank above 7 bar, pull the test lever for 2 seconds. Air must discharge, and the valve must reseat when the lever is released.
2. Replace the valve if it leaks after the test, if the seal is broken, or after 5 years in service.
3. Do not repair or adjust the valve.

## Specifications

### Compressor rating data

| Parameter | Value |
|---|---|
| Compressor type | Single-stage, oil-flooded rotary screw |
| Motor | 15 kW, 2-pole, 2,950 rpm, TEFC |
| Maximum working pressure | 8 bar(g) |
| Free air delivery at 7.5 bar(g) | 2.4 m³/min |
| Drive | 3 V-belts, profile SPA |
| Supply | 400 V, 3-phase, 50 Hz |
| Motor rated current | 27 A at 400 V |
| Starting method | Star-delta |
| Ambient temperature | 2 to 40 °C |
| Oil carryover | Below 3 mg/m³ |

### Lubricant and fill quantities

| Item | Value |
|---|---|
| Compressor oil | ISO VG 46 synthetic compressor oil |
| Total oil capacity | 9.0 L |
| Quantity for an oil change | 8.0 L |
| Thermostatic valve opening temperature | 71 °C |

About 1.0 L of oil stays in the cooler, air end and hoses after draining, which is why an oil change takes 8.0 L. Check the level with the compressor stopped and depressurized for at least 5 minutes. The level must be between the MIN and MAX marks on the sight glass. Do not fill above MAX. Do not mix the synthetic oil with mineral oil, because the mixture foams and raises oil carryover.

### Tightening torques

| Fastener | Size | Torque |
|---|---|---|
| Oil drain plug | G 1/2 | 30 N·m |
| Separator tank cover bolts (8 off) | M12 | 70 N·m |
| Air end and motor mounting bolts | M12 | 85 N·m |
| Pulley taper bush screws | M8 | 20 N·m |
| Oil hose unions | 3/4 in | 60 N·m |
| Oil filter | Spin-on | Hand tight plus 3/4 turn |

The values apply to clean, dry threads. Fit a new sealing ring each time the oil drain plug is removed.

<!-- page: 3 -->

## Installation

### Compressor room and ventilation

Install the compressor indoors on a floor that is flat within 5 mm per meter. The cabinet stands on anti-vibration pads and needs no foundation bolts. Leave 1.0 m free in front of the control panel, 0.8 m at each side, 0.5 m at the rear and 1.0 m above the cooling air outlet.

About 16 kW of heat leaves the cabinet in the cooling air at full load. Ventilate the room with at least 4,800 m³/h of outside air to limit the room temperature rise to 10 °C, or duct the hot air outside through a duct no longer than 3 m.

Below 2 °C the condensate can freeze in the drain. Above 40 °C the discharge temperature reaches the shutdown limit.

### Power supply and cable sizing

| Item | Requirement |
|---|---|
| Supply voltage | 400 V, tolerance 360 to 440 V |
| Voltage imbalance between phases | 2% maximum |
| Supply fuses | 50 A, time-delay |
| Supply cable, up to 50 m | 4 cores, 10 mm² copper |
| Overload relay setting | 15.7 A |

The overload relay sits in the delta branch of the star-delta starter and carries 0.58 times the line current. Its setting is therefore 0.58 times the 27 A rated current, or 15.7 A.

## Operation

### Controller start and stop sequence

1. Close the main isolator. The controller runs a 10 second self-test and displays READY.
2. Press START. The motor starts in star and changes to delta after 6 seconds, with the inlet valve closed.
3. The inlet valve opens and the compressor loads until the network reaches the unload pressure.
4. At the unload pressure the inlet valve closes and the blowdown valve lowers the tank pressure to 2.0 bar. The motor runs on unloaded.
5. After 5 minutes unloaded, the motor stops and the display shows AUTO STANDBY.
6. In AUTO STANDBY the compressor restarts without warning when the pressure falls to the load pressure.
7. Press STOP for a normal stop. The compressor unloads, runs unloaded for 30 seconds and stops.

<!-- page: 4 -->

The controller allows 10 motor starts per hour, at least 6 minutes apart. The emergency stop cuts power under load and floods the separator element with oil. Use it only in an emergency.

### Load and unload pressure settings

| Setting | Factory value | Adjustable range |
|---|---|---|
| Load pressure | 7.0 bar | 5.0 to 7.5 bar |
| Unload pressure | 8.0 bar | 5.5 to 8.0 bar |
| Difference between load and unload | 1.0 bar | 0.5 to 1.5 bar |
| High pressure shutdown (E07) | 8.6 bar | Fixed |
| Minimum pressure valve opening | 4.5 bar | Fixed |

Enter the level 1 password, then edit parameter P02 for the load pressure and P03 for the unload pressure. The controller rejects an unload pressure above 8.0 bar. Each 1 bar reduction of the unload pressure lowers the power consumption by about 7%.

The minimum pressure valve holds the tank at 4.5 bar or above while loaded. This pressure circulates the oil and lets the separator element work.

### Condensate drain operation

The water separator after the aftercooler has a timed solenoid drain. The factory setting opens it for 3 seconds every 4 minutes.

- Press the TEST button on the drain once a day and confirm that condensate discharges.
- The condensate contains oil. Route it to an oil and water separator, never to a surface drain.

The discharge temperature must stay at or above 70 °C. Below that, water condenses in the oil and the oil turns milky.

## Maintenance

### Service interval table

| Interval | Task | Part number |
|---|---|---|
| Daily | Check oil level and controller warnings, test the condensate drain | |
| 500 operating hours | Check V-belt tension, clean the cooler if dust is visible | |
| 2,000 operating hours | Replace the air filter element | EC-15-110 |
| 4,000 operating hours or 12 months | Change the oil, replace the oil filter and the separator element | EC-15-120, EC-15-130 |
| 8,000 operating hours | Replace the V-belts as a set, check the minimum pressure valve | EC-15-140 |

The controller shows warning A03 when an interval is reached. In dusty rooms, halve the air filter interval to 1,000 operating hours.

<!-- page: 5 -->

### Oil and oil filter change

1. Run the compressor loaded until the oil reaches at least 60 °C, so that the oil drains fully.
2. Stop, isolate and depressurize the compressor as described in the isolation and depressurization procedure.
3. Loosen the oil filler plug one turn, wait for any residual pressure to escape, then remove it.
4. Open the tank drain valve and the cooler drain plug, and drain the oil into a container of at least 12 L.
5. Remove the oil filter (EC-15-120) with a filter wrench.
6. Oil the gasket of the new filter. Screw it on until the gasket touches the seat, then a further 3/4 turn by hand.
7. Close the tank drain valve. Refit the cooler drain plug with a new sealing ring at 30 N·m.
8. Fill 8.0 L of ISO VG 46 synthetic compressor oil and refit the filler plug.
9. Run loaded for 5 minutes, stop, wait 5 minutes and top up to the MAX mark.
10. Reset the service counter in menu S01.

### Separator element replacement

Replace the separator element (EC-15-130) every 4,000 operating hours, or earlier when its differential pressure exceeds 1.0 bar.

1. Stop, isolate and depressurize the compressor. Confirm that the tank gauge reads 0 bar.
2. Disconnect the scavenge line and the outlet pipe from the tank cover.
3. Remove the 8 M12 cover bolts and lift the cover off.
4. Lift out the old element and remove both gaskets. Clean the sealing faces.
5. Fit the new element with new gaskets. A metal staple in each gasket earths the element to the tank. Do not remove the staples.
6. Refit the cover and tighten the bolts in a cross pattern to 70 N·m.
7. Set the scavenge tube so that its end is 5 mm above the bottom of the element. A tube that touches the bottom blocks the oil return.
8. Reconnect the pipes and check the cover joint for leaks at 8 bar.

<!-- page: 6 -->

### Drive belt tensioning

The air end is driven by 3 SPA V-belts (set EC-15-140). Check the tension every 500 operating hours.

1. Isolate the compressor and remove the belt guard.
2. Apply 25 N at the middle of each belt span with a tension tester. The deflection must be 6 to 7 mm.
3. To adjust, loosen the 4 motor mounting bolts, turn the tension screw, then tighten the bolts to 85 N·m.
4. Refit the belt guard before you restore power.

Tension new belts to 6 mm and check them again after 50 operating hours, because new belts stretch. Replace all 3 belts together.

## Troubleshooting

### Controller alarm codes

Warnings (A codes) let the compressor run. Shutdowns (E codes) stop it. Press RESET after the cause is cleared.

| Code | Type | Meaning | Corrective action |
|---|---|---|---|
| A01 | Warning | Discharge temperature above 105 °C | See high discharge temperature |
| A02 | Warning | Air filter differential above 50 mbar | Replace air filter element EC-15-110 |
| A03 | Warning | Service interval reached | Carry out the service and reset the counter |
| A04 | Warning | Separator differential above 1.0 bar | Replace separator element EC-15-130 |
| E01 | Shutdown | Discharge temperature reached 110 °C | Let the compressor cool and find the cause |
| E02 | Shutdown | Motor overload relay tripped | Measure motor current and belt tension, reset the relay |
| E03 | Shutdown | Wrong phase sequence or phase loss | Check fuses. Swap two supply phases if the sequence is wrong. |
| E06 | Shutdown | Pressure transducer signal outside 4 to 20 mA | Check the transducer cable and connector. Replace transducer EC-15-410 if the signal stays out of range. |
| E07 | Shutdown | Network pressure above 8.6 bar | Check the inlet valve and the unload solenoid |

### High discharge temperature

The normal discharge temperature is 75 to 95 °C.

<!-- page: 7 -->

| Possible cause | Check | Corrective action |
|---|---|---|
| Low oil level | Level below MIN | Top up with ISO VG 46 synthetic compressor oil |
| Cooler fins blocked | Dust or lint on the cooler face | Blow out against the normal air flow direction with air at no more than 3 bar |
| Room too hot or hot air recirculates | Room temperature above 40 °C | Improve ventilation or duct the cooling air outlet |
| Thermostatic valve stuck in bypass | Cooler outlet hose stays cold while the tank is hot | Replace thermostatic element EC-15-160 |

### Oil carryover in compressed air

Oil in the condensate, oil at air tools or oil consumption above 0.1 L per 100 operating hours indicate carryover above the 3 mg/m³ limit.

| Possible cause | Check | Corrective action |
|---|---|---|
| Oil level above MAX | Sight glass | Drain oil to the MAX mark |
| Scavenge line blocked | No oil flow in the scavenge sight glass when loaded | Clean scavenge orifice EC-15-135, bore 1.2 mm |
| Separator element flooded or damaged | Recent emergency stop under load, or differential below 0.1 bar | Replace the separator element |
| Minimum pressure valve opens early | Tank below 4.5 bar while loaded | Replace minimum pressure valve EC-15-320 |
