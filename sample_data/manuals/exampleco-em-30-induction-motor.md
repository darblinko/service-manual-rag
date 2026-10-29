---
oem: ExampleCo
model: EM-30
title: EM-30 Three-Phase TEFC Induction Motor Service Manual
license: Apache-2.0. Original synthetic text written for this repository.
---
<!-- page: 1 -->

# EM-30 Three-Phase TEFC Induction Motor Service Manual

This is a synthetic manual written for the service-manual-rag sample corpus. The manufacturer, equipment, part numbers and values are fictional. Do not use it to service real equipment.

## Safety

### Electrical isolation and lockout

The terminal box carries the 400 V supply, a separate 230 V supply for the anti-condensation heater and the thermistor leads. Isolate all of them before you open the terminal box or remove the fan cover.

1. Stop the driven machine and the motor from the normal control station.
2. Open the motor isolator. Fit a personal padlock and a tag with your name, the date and the reason for isolation.
3. Isolate the heater supply at its own circuit breaker and lock it.
4. If a variable speed drive feeds the motor, wait at least 10 minutes after isolation for the drive DC link capacitors to discharge.
5. Prove that no voltage is present at U1, V1, W1 and at the heater terminals HE1 and HE2. Test the voltage indicator on a known live source before and after the check.
6. Secure the driven machine against rotation. A fan turned by draft or a pump with reverse flow can turn the motor shaft.

Keep the coupling guard and fan cover fitted whenever the motor can be energized. The frame reaches 80 °C at full load. Do not touch it for 30 minutes after a stop.

### Lifting and handling the motor

The EM-30 weighs 235 kg. Lift it with both eyebolts and a spreader beam so that the slings hang vertically. Without a spreader beam, keep each sling within 30° of vertical.

- Screw each eyebolt fully down to its shoulder before lifting.
- The eyebolts are rated for the motor alone. Do not lift the motor with a pump, gearbox or baseplate attached.
- Remove the transport locking device from the drive end shaft before installation. Refit it whenever the motor is transported.

<!-- page: 2 -->

## Specifications

### Electrical rating plate data

| Parameter | Value |
|---|---|
| Rated output | 30 kW |
| Rated voltage and frequency | 400 V delta or 690 V star, 50 Hz |
| Rated current | 55 A at 400 V |
| Rated speed | 1,475 rpm |
| Number of poles | 4 |
| Power factor at rated load | 0.85 |
| Efficiency at rated load | 93.6% |
| Rated torque | 194 N·m |
| Starting current, direct on line | 7.0 times rated current |
| Insulation class | F, with temperature rise within class B |
| Enclosure | TEFC, IP55 |
| Ambient temperature | -20 to 40 °C, up to 1,000 m altitude |

### Mechanical and bearing data

| Parameter | Value |
|---|---|
| Frame size | 200L, foot mounted |
| Shaft extension | 55 mm diameter, 110 mm long, M20 tapped center hole |
| Drive end bearing | 6312 C3, regreasable, kit EC-30-201 |
| Non-drive end bearing | 6212 C3, regreasable, kit EC-30-202 |
| Bearing grease | Lithium complex NLGI 2 grease, mineral base oil |

### Terminal and fastener torques

| Fastener | Size | Torque |
|---|---|---|
| Main terminal nuts | M8 | 10 N·m |
| Earth terminal in the terminal box | M10 | 20 N·m |
| Terminal box cover screws | M5 | 3 N·m |
| Bearing cover bolts | M8 | 20 N·m |
| End shield bolts | M12 | 60 N·m |
| Foundation bolts | M16 | 170 N·m |

## Installation

### Mounting and shaft alignment

The four foot pads must lie in one plane within 0.1 mm. Use stainless steel shims, with no more than 4 shims under any foot.

1. Set the motor on the base and tighten the foundation bolts by hand.
2. Check soft foot. Place a dial indicator on each foot in turn and loosen that foot's bolt. A rise of more than 0.05 mm indicates soft foot. Correct it with shims.
3. Align the motor to the driven machine to the tolerances in the table.
4. Tighten the foundation bolts to 170 N·m and check the alignment again.

| Parameter | Tolerance at 1,475 rpm |
|---|---|
| Parallel offset | 0.05 mm |
| Angular misalignment | 0.05 mm per 100 mm of coupling diameter |
| Axial gap between coupling hubs | 2.5 to 3.5 mm |

<!-- page: 3 -->

### Terminal box connections

| Supply and starter | Connection | Links |
|---|---|---|
| 400 V, direct on line | Delta | U1 to W2, V1 to U2, W1 to V2 |
| 690 V, direct on line | Star | U2, V2 and W2 joined |
| 400 V, star-delta starter | Six leads to the starter | No links fitted |

- Use a supply cable with 4 cores of 16 mm² copper for runs up to 50 m.
- Clamp each cable lug between the two washers on the stud and tighten the nut to 10 N·m.
- The thermistor leads end at TP1 and TP2. The 230 V, 50 W heater ends at HE1 and HE2.
- Refit the cover gasket (EC-30-405) and tighten the cover screws to 3 N·m.

With L1 on U1, L2 on V1 and L3 on W1, the shaft turns clockwise viewed from the drive end. Swap any two supply conductors to reverse it.

### Insulation resistance test before first start

Test before the first start, after storage longer than 3 months and after any exposure to moisture.

1. Isolate the motor and disconnect the supply cable from the terminals.
2. Do not connect the tester to TP1 and TP2. The thermistors are damaged by voltages above 2.5 V.
3. With the delta links fitted, apply 500 V DC between any main terminal and earth for 1 minute. Record the reading and the winding temperature.
4. Discharge the winding to earth for at least 4 minutes after the test.

The minimum acceptable value is 10 MΩ corrected to 25 °C. The insulation resistance halves for each 10 °C rise in winding temperature, so a reading of 5 MΩ at 35 °C equals 10 MΩ at 25 °C.

Below 10 MΩ, energize the heater for 24 hours to dry the winding and test again. Do not start the motor below 10 MΩ.

## Operation

### Starting methods and starting limits

| Method | Starting current | Starting torque |
|---|---|---|
| Direct on line | 385 A | 446 N·m |
| Star-delta | 128 A | 149 N·m |

<!-- page: 4 -->

With star-delta starting, the load torque during acceleration must stay below 149 N·m. Set the star-delta timer to change over at about 95% of rated speed, typically after 6 to 10 seconds.

Starting limits apply to a load inertia up to 2.0 kg·m².

- No more than 6 starts per hour, evenly spaced.
- No more than 2 consecutive starts from cold, or 1 start from hot operating temperature.
- After consecutive starts, wait 30 minutes before the next start.

On a variable speed drive the permitted frequency range is 5 to 50 Hz. The shaft fan moves less air at low speed. Limit the continuous torque to 80% of rated torque at 25 Hz and 50% at 5 Hz, or fit a separately powered cooling fan.

### Supply voltage imbalance limits

The motor delivers rated output with the voltage within 380 to 420 V and the frequency within 49 to 51 Hz.

Calculate the voltage imbalance as the largest deviation of a line voltage from the three-phase average, divided by the average, times 100. For example, 400 V, 404 V and 396 V give an average of 400 V and a largest deviation of 4 V, which is 1.0%.

| Voltage imbalance | Permitted output |
|---|---|
| Up to 1% | 30 kW (100%) |
| 2% | 28.5 kW (95%) |
| 3% | 26.4 kW (88%) |
| Above 3% | Do not operate |

### Thermistor protection

Three PTC thermistors, one in each phase winding, are wired in series to TP1 and TP2. Their response temperature is 150 °C. Connect them to a thermistor relay that opens the contactor coil circuit.

| Thermistor chain condition | Resistance |
|---|---|
| Cold, at 20 °C | 60 to 750 Ω |
| Relay trip value | Above 3,600 Ω |
| Relay reset value | Below 1,800 Ω |

- Measure the chain with an ohmmeter whose test voltage does not exceed 2.5 V.
- Set the relay to manual reset. After a trip the winding needs 20 to 40 minutes to cool below the reset value.

<!-- page: 5 -->

## Maintenance

### Inspection schedule

| Interval | Task |
|---|---|
| Weekly | Listen for bearing noise. Measure vibration on both bearing housings. |
| 2,000 operating hours | Clean the fan cover intake and cooling fins. Check foundation bolt tightness. |
| 8,000 operating hours or 2 years | Regrease both bearings. |
| 8,000 operating hours or annually | Test insulation resistance. Check the main terminal nuts at 10 N·m. |
| 40,000 operating hours | Replace both bearings. |

### Bearing regreasing procedure

Regrease both bearings every 8,000 operating hours or every 2 years, whichever comes first. The interval applies to a horizontal shaft at 1,475 rpm and an ambient temperature up to 40 °C. Halve it for a vertical shaft or a higher ambient temperature.

| Bearing | Type | Grease quantity |
|---|---|---|
| Drive end | 6312 C3 | 20 g |
| Non-drive end | 6212 C3 | 15 g |

Use lithium complex NLGI 2 grease with a base oil viscosity of 100 to 150 mm²/s at 40 °C.

1. Clean the grease nipples and open the grease outlet at the bottom of each end shield.
2. With the motor running, inject 20 g at the drive end and 15 g at the non-drive end. If greasing while running is not possible, inject half the quantity at standstill, run for 1 hour and inject the rest.
3. Run the motor for 2 hours with the outlets open, then close them.

The bearing temperature can rise by up to 15 °C after regreasing and must return to normal within 8 hours.

### Bearing replacement

Replace both bearings every 40,000 operating hours, or earlier when noise or vibration indicates bearing damage.

<!-- page: 6 -->

1. Isolate the motor as described under electrical isolation and lockout. Uncouple it and pull off the coupling hub.
2. Remove the fan cover, the circlip and the fan (EC-30-310). Remove the bearing cover bolts and end shield bolts at both ends.
3. Support the rotor with a sling and withdraw it toward the drive end without touching the stator winding.
4. Pull each bearing off with a puller that grips the inner ring.
5. Heat each new bearing in an induction heater to 110 °C. Do not exceed 120 °C or use an open flame.
6. Slide each bearing against its shaft shoulder and hold it for 10 seconds until it grips.
7. When cool, pack each bearing fully with grease and fill the bearing cover cavities to 50%.
8. Reassemble. Tighten the end shield bolts to 60 N·m in a cross pattern and the bearing cover bolts to 20 N·m.
9. Turn the shaft by hand to confirm it turns without rubbing.
10. Run the motor uncoupled for 1 hour. Vibration must be below 2.8 mm/s and the bearing housing must stabilize below 80 °C.

## Troubleshooting

### Motor fails to start or hums

| Symptom | Possible cause | Corrective action |
|---|---|---|
| No sound when started | No supply, open fuse or thermistor relay tripped | Check fuses, isolator and thermistor relay |
| Hums and does not turn | One supply phase missing | Measure voltage on U1, V1 and W1. Check fuses and contactor contacts. |
| Turns slowly in star, trips at changeover | Load torque above the 149 N·m star starting torque | Reduce the load during the start or use direct on line starting |
| Current at locked rotor value, shaft does not turn | Seized bearing or blocked driven machine | Isolate and turn the shaft by hand. Check the driven machine. |

<!-- page: 7 -->

### Overheating and thermistor trips

| Possible cause | Check | Corrective action |
|---|---|---|
| Overload | Line current above 55 A | Reduce the load on the driven machine |
| Blocked cooling | Dust on fins, intake closer than 100 mm to a wall | Clean the fins and clear the intake |
| Supply out of tolerance | Voltage outside 380 to 420 V, or imbalance above 1% | Correct the supply or derate the motor |
| Faulty thermistor circuit | Trip with normal current and a cool frame, cold chain resistance outside 60 to 750 Ω | Repair the thermistor leads or replace the relay |

### Bearing noise and vibration

Measure RMS vibration velocity on both bearing housings in the horizontal, vertical and axial directions, from 10 to 1,000 Hz.

| Condition | RMS vibration velocity |
|---|---|
| Acceptance limit, new or rebuilt motor | 2.8 mm/s |
| Alarm | 4.5 mm/s |
| Shutdown | 7.1 mm/s |

| Symptom | Possible cause | Corrective action |
|---|---|---|
| High axial vibration at 1 and 2 times running speed | Misalignment | Realign to 0.05 mm offset |
| High radial vibration at 24.6 Hz | Unbalance in the coupling or key | Check the coupling balance. The rotor is balanced with a half key. |
| Rumbling or grinding, rising bearing temperature | Bearing damage | Replace both bearings |
| Whining noise on a motor fed by a drive | Bearing current damage | Fit an insulated non-drive end bearing, kit EC-30-215 |
