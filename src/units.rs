#[derive(Debug, Clone)]
pub enum Unit {
    Mass(Mass),
    Time(Time),
    Length(Length),
    Speed(Speed),
    Angle(Angle),
    Pressure(Pressure),
    Area(Area),
    Volume(Volume),
    Temperature(Temperature),
    Energy(Energy),
    Power(Power),
    Force(Force),
    ElectricCharge(ElectricCharge),
    Current(Current),
    Frequency(Frequency),
    Acceleration(Acceleration),
}

#[derive(Debug, Clone)]
pub enum Mass {
    Kg,
    Gram,
    Ton,
    Grain,
    Carat,
    Pound,
    Ounce,
    Atomic,
    Earth,
    Sun,
    Moon,
    Jupiter,
    Milligram,
    Microgram,
    Nanogram,
    Custom(f64, Box<Mass>),
}

impl Mass {
    pub fn convert_to(self, to: Self) -> Self {
        let base_kg = match self {
            Mass::Kg => 1.0,
            Mass::Gram => 0.001,
            Mass::Ton => 1000.0,
            Mass::Grain => 0.0000647989,
            Mass::Carat => 0.0002,
            Mass::Pound => 0.45359237,
            Mass::Ounce => 0.028349523125,
            Mass::Atomic => 1.660539067e-27,
            Mass::Earth => 5.972e24,
            Mass::Sun => 1.989e30,
            Mass::Moon => 7.34767309e22,
            Mass::Jupiter => 1.8982e27,
            Mass::Milligram => 0.000001,
            Mass::Microgram => 0.000000001,
            Mass::Nanogram => 0.000000000001,
            Mass::Custom(value, base) => {
                value
                    * match *base {
                        Mass::Kg => 1.0,
                        Mass::Gram => 0.001,
                        Mass::Ton => 1000.0,
                        Mass::Grain => 0.0000647989,
                        Mass::Carat => 0.0002,
                        Mass::Pound => 0.45359237,
                        Mass::Ounce => 0.028349523125,
                        Mass::Atomic => 1.660539067e-27,
                        Mass::Earth => 5.972e24,
                        Mass::Sun => 1.989e30,
                        Mass::Moon => 7.34767309e22,
                        Mass::Jupiter => 1.8982e27,
                        Mass::Milligram => 0.000001,
                        Mass::Microgram => 0.000000001,
                        Mass::Nanogram => 0.000000000001,
                        Mass::Custom(v, b) => v * b.convert_to(Mass::Kg).to_kg(),
                    }
            }
        };

        match to {
            Mass::Kg => Mass::Kg,
            Mass::Gram => Mass::Custom(base_kg / 0.001, Box::new(Mass::Gram)),
            Mass::Ton => Mass::Custom(base_kg / 1000.0, Box::new(Mass::Ton)),
            Mass::Grain => Mass::Custom(base_kg / 0.0000647989, Box::new(Mass::Grain)),
            Mass::Carat => Mass::Custom(base_kg / 0.0002, Box::new(Mass::Carat)),
            Mass::Pound => Mass::Custom(base_kg / 0.45359237, Box::new(Mass::Pound)),
            Mass::Ounce => Mass::Custom(base_kg / 0.028349523125, Box::new(Mass::Ounce)),
            Mass::Atomic => Mass::Custom(base_kg / 1.660539067e-27, Box::new(Mass::Atomic)),
            Mass::Earth => Mass::Custom(base_kg / 5.972e24, Box::new(Mass::Earth)),
            Mass::Sun => Mass::Custom(base_kg / 1.989e30, Box::new(Mass::Sun)),
            Mass::Moon => Mass::Custom(base_kg / 7.34767309e22, Box::new(Mass::Moon)),
            Mass::Jupiter => Mass::Custom(base_kg / 1.8982e27, Box::new(Mass::Jupiter)),
            Mass::Milligram => Mass::Custom(base_kg / 0.000001, Box::new(Mass::Milligram)),
            Mass::Microgram => Mass::Custom(base_kg / 0.000000001, Box::new(Mass::Microgram)),
            Mass::Nanogram => Mass::Custom(base_kg / 0.000000000001, Box::new(Mass::Nanogram)),
            Mass::Custom(value, base) => {
                let base_in_kg = base.clone().convert_to(Mass::Kg).to_kg();
                Mass::Custom(base_kg / (value * base_in_kg), base)
            }
        }
    }

    pub fn to_kg(&self) -> f64 {
        match self {
            Mass::Kg => 1.0,
            Mass::Gram => 0.001,
            Mass::Ton => 1000.0,
            Mass::Grain => 0.0000647989,
            Mass::Carat => 0.0002,
            Mass::Pound => 0.45359237,
            Mass::Ounce => 0.028349523125,
            Mass::Atomic => 1.660539067e-27,
            Mass::Earth => 5.972e24,
            Mass::Sun => 1.989e30,
            Mass::Moon => 7.34767309e22,
            Mass::Jupiter => 1.8982e27,
            Mass::Milligram => 0.000001,
            Mass::Microgram => 0.000000001,
            Mass::Nanogram => 0.000000000001,
            Mass::Custom(value, base) => value * base.to_kg(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Time {
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    Minute,
    Hour,
    Day,
    Week,
    Year,
    JulianYear,
    Decade,
    Century,
    SiderealDay,
    SiderealYear,
    Custom(f64, Box<Time>),
}

impl Time {
    pub fn convert_to(self, to: Self) -> Self {
        let base_seconds = match self {
            Time::Second => 1.0,
            Time::Millisecond => 0.001,
            Time::Microsecond => 0.000001,
            Time::Nanosecond => 0.000000001,
            Time::Minute => 60.0,
            Time::Hour => 3600.0,
            Time::Day => 86400.0,
            Time::Week => 604800.0,
            Time::Year => 31536000.0,
            Time::JulianYear => 31557600.0,
            Time::Decade => 315360000.0,
            Time::Century => 3153600000.0,
            Time::SiderealDay => 86164.0905,
            Time::SiderealYear => 31558149.504,
            Time::Custom(value, base) => {
                value
                    * match *base {
                        Time::Second => 1.0,
                        Time::Millisecond => 0.001,
                        Time::Microsecond => 0.000001,
                        Time::Nanosecond => 0.000000001,
                        Time::Minute => 60.0,
                        Time::Hour => 3600.0,
                        Time::Day => 86400.0,
                        Time::Week => 604800.0,
                        Time::Year => 31536000.0,
                        Time::JulianYear => 31557600.0,
                        Time::Decade => 315360000.0,
                        Time::Century => 3153600000.0,
                        Time::SiderealDay => 86164.0905,
                        Time::SiderealYear => 31558149.504,
                        Time::Custom(v, b) => v * b.convert_to(Time::Second).to_seconds(),
                    }
            }
        };

        match to {
            Time::Second => Time::Second,
            Time::Millisecond => Time::Custom(base_seconds / 0.001, Box::new(Time::Millisecond)),
            Time::Microsecond => Time::Custom(base_seconds / 0.000001, Box::new(Time::Microsecond)),
            Time::Nanosecond => {
                Time::Custom(base_seconds / 0.000000001, Box::new(Time::Nanosecond))
            }
            Time::Minute => Time::Custom(base_seconds / 60.0, Box::new(Time::Minute)),
            Time::Hour => Time::Custom(base_seconds / 3600.0, Box::new(Time::Hour)),
            Time::Day => Time::Custom(base_seconds / 86400.0, Box::new(Time::Day)),
            Time::Week => Time::Custom(base_seconds / 604800.0, Box::new(Time::Week)),
            Time::Year => Time::Custom(base_seconds / 31536000.0, Box::new(Time::Year)),
            Time::JulianYear => Time::Custom(base_seconds / 31557600.0, Box::new(Time::JulianYear)),
            Time::Decade => Time::Custom(base_seconds / 315360000.0, Box::new(Time::Decade)),
            Time::Century => Time::Custom(base_seconds / 3153600000.0, Box::new(Time::Century)),
            Time::SiderealDay => {
                Time::Custom(base_seconds / 86164.0905, Box::new(Time::SiderealDay))
            }
            Time::SiderealYear => {
                Time::Custom(base_seconds / 31558149.504, Box::new(Time::SiderealYear))
            }
            Time::Custom(value, base) => {
                let base_in_seconds = base.clone().convert_to(Time::Second).to_seconds();
                Time::Custom(base_seconds / (value * base_in_seconds), base)
            }
        }
    }

    pub fn to_seconds(&self) -> f64 {
        match self {
            Time::Second => 1.0,
            Time::Millisecond => 0.001,
            Time::Microsecond => 0.000001,
            Time::Nanosecond => 0.000000001,
            Time::Minute => 60.0,
            Time::Hour => 3600.0,
            Time::Day => 86400.0,
            Time::Week => 604800.0,
            Time::Year => 31536000.0,
            Time::JulianYear => 31557600.0,
            Time::Decade => 315360000.0,
            Time::Century => 3153600000.0,
            Time::SiderealDay => 86164.0905,
            Time::SiderealYear => 31558149.504,
            Time::Custom(value, base) => value * base.to_seconds(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Length {
    Meter,
    Centimeter,
    Kilometer,
    Inch,
    Foot,
    Yard,
    Mile,
    Mil,
    Point,
    NauticalMile,
    Fermi,
    Micron,
    AstroUnit,
    LightYear,
    LightMinute,
    LightSecond,
    LightDay,
    Parsec,
    Millimeter,
    Micrometer,
    Nanometer,
    Angstrom,
    Megaparsec,
    Gigaparsec,
    Custom(f64, Box<Length>),
}

impl Length {
    pub fn convert_to(self, to: Self) -> Self {
        let base_meters = self.to_meters();

        match to {
            Length::Meter => Length::Meter,
            Length::Centimeter => Length::Custom(base_meters * 100.0, Box::new(Length::Centimeter)),
            Length::Kilometer => Length::Custom(base_meters / 1000.0, Box::new(Length::Kilometer)),
            Length::Inch => Length::Custom(base_meters / 0.0254, Box::new(Length::Inch)),
            Length::Foot => Length::Custom(base_meters / 0.3048, Box::new(Length::Foot)),
            Length::Yard => Length::Custom(base_meters / 0.9144, Box::new(Length::Yard)),
            Length::Mile => Length::Custom(base_meters / 1609.344, Box::new(Length::Mile)),
            Length::Mil => Length::Custom(base_meters / 0.0000254, Box::new(Length::Mil)),
            Length::Point => Length::Custom(base_meters / 0.0003527777778, Box::new(Length::Point)),
            Length::NauticalMile => {
                Length::Custom(base_meters / 1852.0, Box::new(Length::NauticalMile))
            }
            Length::Fermi => Length::Custom(base_meters / 1e-15, Box::new(Length::Fermi)),
            Length::Micron => Length::Custom(base_meters / 0.000001, Box::new(Length::Micron)),
            Length::AstroUnit => {
                Length::Custom(base_meters / 149597870700.0, Box::new(Length::AstroUnit))
            }
            Length::LightYear => {
                Length::Custom(base_meters / 9.461e15, Box::new(Length::LightYear))
            }
            Length::LightMinute => {
                Length::Custom(base_meters / 17987547480.0, Box::new(Length::LightMinute))
            }
            Length::LightSecond => {
                Length::Custom(base_meters / 299792458.0, Box::new(Length::LightSecond))
            }
            Length::LightDay => {
                Length::Custom(base_meters / 25902068371200.0, Box::new(Length::LightDay))
            }
            Length::Parsec => Length::Custom(base_meters / 3.086e16, Box::new(Length::Parsec)),
            Length::Millimeter => Length::Custom(base_meters / 0.001, Box::new(Length::Millimeter)),
            Length::Micrometer => {
                Length::Custom(base_meters / 0.000001, Box::new(Length::Micrometer))
            }
            Length::Nanometer => {
                Length::Custom(base_meters / 0.000000001, Box::new(Length::Nanometer))
            }
            Length::Angstrom => Length::Custom(base_meters / 1e-10, Box::new(Length::Angstrom)),
            Length::Megaparsec => {
                Length::Custom(base_meters / 3.086e22, Box::new(Length::Megaparsec))
            }
            Length::Gigaparsec => {
                Length::Custom(base_meters / 3.086e25, Box::new(Length::Gigaparsec))
            }
            Length::Custom(value, base) => {
                let base_in_meters = base.clone().convert_to(Length::Meter).to_meters();
                Length::Custom(base_meters / (value * base_in_meters), base)
            }
        }
    }

    pub fn to_meters(&self) -> f64 {
        match self {
            Length::Meter => 1.0,
            Length::Centimeter => 0.01,
            Length::Kilometer => 1000.0,
            Length::Inch => 0.0254,
            Length::Foot => 0.3048,
            Length::Yard => 0.9144,
            Length::Mile => 1609.344,
            Length::Mil => 0.0000254,
            Length::Point => 0.0003527777778,
            Length::NauticalMile => 1852.0,
            Length::Fermi => 1e-15,
            Length::Micron => 0.000001,
            Length::AstroUnit => 149597870700.0,
            Length::LightYear => 9.461e15,
            Length::LightMinute => 17987547480.0,
            Length::LightSecond => 299792458.0,
            Length::LightDay => 25902068371200.0,
            Length::Parsec => 3.086e16,
            Length::Millimeter => 0.001,
            Length::Micrometer => 0.000001,
            Length::Nanometer => 0.000000001,
            Length::Angstrom => 1e-10,
            Length::Megaparsec => 3.086e22,
            Length::Gigaparsec => 3.086e25,
            Length::Custom(value, base) => value * base.to_meters(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Speed {
    KmH,
    MpH,
    Mach,
    Knot,
    MetersPerSecond,
    C,
    Custom(Length, Time),
}

impl Speed {
    pub fn convert_to(self, to: Self) -> Self {
        let base_mps = self.to_mps();

        match to {
            Speed::MetersPerSecond => Speed::MetersPerSecond,
            Speed::KmH => {
                let km_per_hour = base_mps * 3.6; // Convert m/s to km/h
                Speed::Custom(
                    Length::Custom(km_per_hour, Box::new(Length::Kilometer)),
                    Time::Hour,
                )
            }
            Speed::MpH => {
                let miles_per_hour = base_mps * 2.23694; // Convert m/s to mph
                Speed::Custom(
                    Length::Custom(miles_per_hour, Box::new(Length::Mile)),
                    Time::Hour,
                )
            }
            Speed::Mach => {
                let mach = base_mps / 343.0; // Convert m/s to Mach number
                Speed::Custom(
                    Length::Custom(mach * 343.0, Box::new(Length::Meter)),
                    Time::Second,
                )
            }
            Speed::Knot => {
                let knots = base_mps * 1.94384; // Convert m/s to knots
                Speed::Custom(
                    Length::Custom(knots * 1852.0, Box::new(Length::NauticalMile)),
                    Time::Hour,
                )
            }
            Speed::C => {
                let c = base_mps / 299792458.0; 
                Speed::Custom(
                    Length::Custom(c * 299792458.0, Box::new(Length::Meter)),
                    Time::Second,
                )
            }
            Speed::Custom(length, time) => {
                let target_length = Length::Meter.convert_to(length);
                let target_time = Time::Second.convert_to(time);
                Speed::Custom(target_length, target_time)
            }
        }
    }

    pub fn to_mps(&self) -> f64 {
        match self {
            Speed::MetersPerSecond => 1.0,
            Speed::KmH => 0.277778,
            Speed::MpH => 0.44704,
            Speed::Mach => 343.0,
            Speed::Knot => 0.514444,
            Speed::C => 299792458.0,
            Speed::Custom(length, time) => {
                let meters = match length {
                    Length::Custom(value, base) => value * base.to_meters(),
                    _ => length.clone().convert_to(Length::Meter).to_meters(),
                };
                let seconds = match time {
                    Time::Custom(value, base) => value * base.to_seconds(),
                    _ => time.clone().convert_to(Time::Second).to_seconds(),
                };
                meters / seconds
            }
        }
    }

    pub fn to_kmh(&self) -> f64 {
        self.to_mps() * 3.6
    }

    pub fn to_mph(&self) -> f64 {
        self.to_mps() * 2.23694
    }

    pub fn to_knots(&self) -> f64 {
        self.to_mps() * 1.94384
    }

    pub fn to_mach(&self) -> f64 {
        self.to_mps() / 343.0
    }

    pub fn to_c(&self) -> f64 {
        self.to_mps() / 299792458.0
    }
}

#[derive(Debug, Clone)]
pub enum Angle {
    Degree,
    Radian,
    Gradian,
    ArcMinute,
    ArcSecond,
    Custom(f64, Box<Angle>),
}

impl Angle {
    pub fn convert_to(self, to: Self) -> Self {
        // Convert to radians as base unit
        let base_radians = self.to_radians();

        match to {
            Angle::Radian => Angle::Radian,
            Angle::Degree => {
                Angle::Custom(base_radians * 57.29577951308232, Box::new(Angle::Degree))
            } // 180/π
            Angle::Gradian => {
                Angle::Custom(base_radians * 63.66197723675813, Box::new(Angle::Gradian))
            } // 200/π
            Angle::ArcMinute => Angle::Custom(
                base_radians * 3437.7467707849396,
                Box::new(Angle::ArcMinute),
            ), // 10800/π
            Angle::ArcSecond => Angle::Custom(
                base_radians * 206264.80624709636,
                Box::new(Angle::ArcSecond),
            ), // 648000/π
            Angle::Custom(value, base) => {
                let base_in_radians = base.clone().convert_to(Angle::Radian).to_radians();
                Angle::Custom(base_radians / (value * base_in_radians), base)
            }
        }
    }

    pub fn to_radians(&self) -> f64 {
        match self {
            Angle::Radian => 1.0,
            Angle::Degree => 0.017453292519943295,
            Angle::Gradian => 0.015707963267948967,
            Angle::ArcMinute => 0.0002908882086657216,
            Angle::ArcSecond => 0.000004848136811095278,
            Angle::Custom(value, base) => value * base.to_radians(),
        }
    }

    pub fn to_degrees(&self) -> f64 {
        self.to_radians() * 57.29577951308232
    }
}

#[derive(Debug, Clone)]
pub enum Pressure {
    Atmospheric,
    Bar,
    Millibar,
    Pascal,
    Kilopascal,
    Megapascal,
    Torr,
    MmHg,
    Psi,
    Custom(Force, Area),
}

impl Pressure {
    pub fn convert_to(self, to: Self) -> Self {
        let base_pascal = self.to_pascal();

        match to {
            Pressure::Pascal => Pressure::Pascal,
            Pressure::Atmospheric => Pressure::Custom(
                Force::Custom(
                    Mass::Custom(base_pascal / 101325.0, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Area::SquareMeter,
            ),
            Pressure::Bar => Pressure::Custom(
                Force::Custom(
                    Mass::Custom(base_pascal / 100000.0, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Area::SquareMeter,
            ),
            Pressure::Millibar => Pressure::Custom(
                Force::Custom(
                    Mass::Custom(base_pascal / 100.0, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Area::SquareMeter,
            ),
            Pressure::Kilopascal => Pressure::Custom(
                Force::Custom(
                    Mass::Custom(base_pascal / 1000.0, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Area::SquareMeter,
            ),
            Pressure::Megapascal => Pressure::Custom(
                Force::Custom(
                    Mass::Custom(base_pascal / 1000000.0, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Area::SquareMeter,
            ),
            Pressure::Torr => Pressure::Custom(
                Force::Custom(
                    Mass::Custom(base_pascal / 133.322, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Area::SquareMeter,
            ),
            Pressure::MmHg => Pressure::Custom(
                Force::Custom(
                    Mass::Custom(base_pascal / 133.322, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Area::SquareMeter,
            ),
            Pressure::Psi => Pressure::Custom(
                Force::Custom(
                    Mass::Custom(base_pascal / 6894.757293168361, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Area::SquareMeter,
            ),
            Pressure::Custom(force, area) => {
                let target_force = Force::Newton.convert_to(force);
                let target_area = Area::SquareMeter.convert_to(area);
                Pressure::Custom(target_force, target_area)
            }
        }
    }

    pub fn to_pascal(&self) -> f64 {
        match self {
            Pressure::Pascal => 1.0,
            Pressure::Atmospheric => 101325.0,
            Pressure::Bar => 100000.0,
            Pressure::Millibar => 100.0,
            Pressure::Kilopascal => 1000.0,
            Pressure::Megapascal => 1000000.0,
            Pressure::Torr => 133.322,
            Pressure::MmHg => 133.322,
            Pressure::Psi => 6894.757293168361,
            Pressure::Custom(force, area) => {
                let newtons = match force {
                    Force::Newton => 1.0,
                    Force::Dyne => 0.00001,
                    Force::Pound => 4.448222,
                    Force::Kg => 9.80665,
                    Force::Custom(mass, length, time) => {
                        let kg = mass.clone().convert_to(Mass::Kg).to_kg();
                        let meters = length.clone().convert_to(Length::Meter).to_meters();
                        let seconds = time.clone().convert_to(Time::Second).to_seconds();
                        kg * (meters / (seconds * seconds))
                    }
                };

                let square_meters = match area {
                    Area::SquareMeter => 1.0,
                    Area::Acre => 4046.8564224,
                    Area::Hectare => 10000.0,
                    Area::SquareKilometer => 1000000.0,
                    Area::SquareMile => 2589988.110336,
                    Area::SquareFoot => 0.09290304,
                    Area::Custom(length) => {
                        let meters = length.clone().convert_to(Length::Meter).to_meters();
                        meters * meters
                    }
                };

                newtons / square_meters
            }
        }
    }

    pub fn to_bar(&self) -> f64 {
        self.to_pascal() / 100000.0
    }

    pub fn to_atm(&self) -> f64 {
        self.to_pascal() / 101325.0
    }

    pub fn to_psi(&self) -> f64 {
        self.to_pascal() / 6894.757293168361
    }
}

#[derive(Debug, Clone)]
pub enum Area {
    Acre,
    Hectare,
    SquareMeter,
    SquareKilometer,
    SquareMile,
    SquareFoot,
    Custom(Length),
}

impl Area {
    pub fn convert_to(self, to: Self) -> Self {
        let base_square_meters = self.to_square_meters();

        match to {
            Area::SquareMeter => Area::Custom(Length::Meter),
            Area::Acre => Area::Custom(Length::Custom(
                (base_square_meters / 4046.8564224).sqrt(),
                Box::new(Length::Custom(2.0, Box::new(Length::Meter))),
            )),
            Area::Hectare => Area::Custom(Length::Custom(
                (base_square_meters / 10000.0).sqrt(),
                Box::new(Length::Custom(100.0, Box::new(Length::Meter))),
            )),
            Area::SquareKilometer => Area::Custom(Length::Custom(
                (base_square_meters / 1_000_000.0).sqrt(),
                Box::new(Length::Kilometer),
            )),
            Area::SquareMile => Area::Custom(Length::Custom(
                (base_square_meters / 2_589_988.110336).sqrt(),
                Box::new(Length::Mile),
            )),
            Area::SquareFoot => Area::Custom(Length::Custom(
                (base_square_meters / 0.09290304).sqrt(),
                Box::new(Length::Foot),
            )),
            Area::Custom(length) => {
                let target_length = Length::Meter.convert_to(length);
                Area::Custom(target_length)
            }
        }
    }

    pub fn to_square_meters(&self) -> f64 {
        match self {
            Area::SquareMeter => 1.0,
            Area::Acre => 4046.8564224,
            Area::Hectare => 10000.0,
            Area::SquareKilometer => 1_000_000.0,
            Area::SquareMile => 2_589_988.110336,
            Area::SquareFoot => 0.09290304,
            Area::Custom(length) => {
                let meters = length.clone().convert_to(Length::Meter).to_meters();
                meters * meters
            }
        }
    }

    pub fn to_acres(&self) -> f64 {
        self.to_square_meters() / 4046.8564224
    }

    pub fn to_hectares(&self) -> f64 {
        self.to_square_meters() / 10000.0
    }

    pub fn to_square_kilometers(&self) -> f64 {
        self.to_square_meters() / 1_000_000.0
    }

    pub fn to_square_miles(&self) -> f64 {
        self.to_square_meters() / 2_589_988.110336
    }

    pub fn to_square_feet(&self) -> f64 {
        self.to_square_meters() / 0.09290304
    }
}

impl Length {
    pub fn squared(&self) -> Area {
        Area::Custom(self.clone())
    }
}

#[derive(Debug, Clone)]
pub enum Volume {
    Custom(Length),
    CubicMeter,
    CubicFoot,
    Liter,
    Milliliter,
    Gallon,
    Barell,
    FluidOunce,
    Pint,
    Quart,
}

impl Volume {
    pub fn convert_to(self, to: Self) -> Self {
        let base_cubic_meters = self.to_cubic_meters();

        match to {
            Volume::CubicMeter => Volume::Custom(Length::Meter),
            Volume::CubicFoot => Volume::Custom(Length::Custom(
                (base_cubic_meters / 0.028316846592).cbrt(),
                Box::new(Length::Foot),
            )),
            Volume::Liter => Volume::Custom(Length::Custom(
                (base_cubic_meters / 0.001).cbrt(),
                Box::new(Length::Custom(0.1, Box::new(Length::Meter))),
            )),
            Volume::Milliliter => Volume::Custom(Length::Custom(
                (base_cubic_meters / 0.000001).cbrt(),
                Box::new(Length::Custom(0.01, Box::new(Length::Meter))),
            )),
            Volume::Gallon => Volume::Custom(Length::Custom(
                (base_cubic_meters / 0.003785411784).cbrt(),
                Box::new(Length::Custom(0.1546, Box::new(Length::Meter))),
            )),
            Volume::Barell => Volume::Custom(Length::Custom(
                (base_cubic_meters / 0.158987294928).cbrt(),
                Box::new(Length::Custom(0.5408, Box::new(Length::Meter))),
            )),
            Volume::FluidOunce => Volume::Custom(Length::Custom(
                (base_cubic_meters / 0.0000295735295625).cbrt(),
                Box::new(Length::Custom(0.0312, Box::new(Length::Meter))),
            )),
            Volume::Pint => Volume::Custom(Length::Custom(
                (base_cubic_meters / 0.000473176473).cbrt(),
                Box::new(Length::Custom(0.0781, Box::new(Length::Meter))),
            )),
            Volume::Quart => Volume::Custom(Length::Custom(
                (base_cubic_meters / 0.000946352946).cbrt(),
                Box::new(Length::Custom(0.0984, Box::new(Length::Meter))),
            )),
            Volume::Custom(length) => {
                let target_length = Length::Meter.convert_to(length);
                Volume::Custom(target_length)
            }
        }
    }

    pub fn to_cubic_meters(&self) -> f64 {
        match self {
            Volume::CubicMeter => 1.0,
            Volume::CubicFoot => 0.028316846592,
            Volume::Liter => 0.001,
            Volume::Milliliter => 0.000001,
            Volume::Gallon => 0.003785411784,
            Volume::Barell => 0.158987294928,
            Volume::FluidOunce => 0.0000295735295625,
            Volume::Pint => 0.000473176473,
            Volume::Quart => 0.000946352946,
            Volume::Custom(length) => {
                let meters = length.clone().convert_to(Length::Meter).to_meters();
                meters * meters * meters
            }
        }
    }

    pub fn to_liters(&self) -> f64 {
        self.to_cubic_meters() * 1000.0
    }

    pub fn to_gallons(&self) -> f64 {
        self.to_cubic_meters() / 0.003785411784
    }

    pub fn to_cubic_feet(&self) -> f64 {
        self.to_cubic_meters() / 0.028316846592
    }

    pub fn to_milliliters(&self) -> f64 {
        self.to_cubic_meters() * 1_000_000.0
    }

    pub fn to_fluid_ounces(&self) -> f64 {
        self.to_cubic_meters() / 0.0000295735295625
    }

    pub fn to_pints(&self) -> f64 {
        self.to_cubic_meters() / 0.000473176473
    }

    pub fn to_quarts(&self) -> f64 {
        self.to_cubic_meters() / 0.000946352946
    }

    pub fn to_barrels(&self) -> f64 {
        self.to_cubic_meters() / 0.158987294928
    }
}

impl Length {
    pub fn cubed(&self) -> Volume {
        Volume::Custom(self.clone())
    }
}

#[derive(Debug, Clone)]
pub enum Temperature {
    Celsius,
    Fahrenheit,
    Kelvin,
    Rankine,
}

impl Temperature {
    pub fn convert_to(self, to: Self) -> Self {
        match to {
            Temperature::Kelvin => Temperature::Kelvin,
            Temperature::Celsius => Temperature::Celsius,
            Temperature::Fahrenheit => Temperature::Fahrenheit,
            Temperature::Rankine => Temperature::Rankine,
        }
    }

    pub fn to_kelvin(&self, value: f64) -> f64 {
        match self {
            Temperature::Kelvin => value,
            Temperature::Celsius => value + 273.15,
            Temperature::Fahrenheit => (value - 32.0) * 5.0 / 9.0 + 273.15,
            Temperature::Rankine => value * 5.0 / 9.0,
        }
    }

    pub fn from_kelvin(&self, kelvin: f64) -> f64 {
        match self {
            Temperature::Kelvin => kelvin,
            Temperature::Celsius => kelvin - 273.15,
            Temperature::Fahrenheit => (kelvin - 273.15) * 9.0 / 5.0 + 32.0,
            Temperature::Rankine => kelvin * 9.0 / 5.0,
        }
    }

    pub fn to_celsius(&self, value: f64) -> f64 {
        let kelvin = self.to_kelvin(value);
        Temperature::Celsius.from_kelvin(kelvin)
    }

    pub fn to_fahrenheit(&self, value: f64) -> f64 {
        let kelvin = self.to_kelvin(value);
        Temperature::Fahrenheit.from_kelvin(kelvin)
    }

    pub fn to_rankine(&self, value: f64) -> f64 {
        let kelvin = self.to_kelvin(value);
        Temperature::Rankine.from_kelvin(kelvin)
    }
}

#[derive(Debug, Clone)]
pub enum Energy {
    Joule,
    Kilojoule,
    Megajoule,
    Ev,
    KiloEv,
    MegaEv,
    Calorie,
    Kilocalorie,
    Erg,
    TNT,
    // Custom energy as work
    Custom(Force, Length),
}

impl Energy {
    pub fn convert_to(self, to: Self) -> Self {
        let base_joules = self.to_joules();

        match to {
            Energy::Joule => Energy::Joule,
            Energy::Kilojoule => Energy::Custom(
                Force::Custom(
                    Mass::Custom(base_joules / 1000.0, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Length::Meter,
            ),
            Energy::Megajoule => Energy::Custom(
                Force::Custom(
                    Mass::Custom(base_joules / 1_000_000.0, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Length::Meter,
            ),
            Energy::Ev => Energy::Custom(
                Force::Custom(
                    Mass::Custom(base_joules / 1.602176634e-19, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Length::Meter,
            ),
            Energy::KiloEv => Energy::Custom(
                Force::Custom(
                    Mass::Custom(base_joules / 1.602176634e-16, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Length::Meter,
            ),
            Energy::MegaEv => Energy::Custom(
                Force::Custom(
                    Mass::Custom(base_joules / 1.602176634e-13, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Length::Meter,
            ),
            Energy::Calorie => Energy::Custom(
                Force::Custom(
                    Mass::Custom(base_joules / 4.184, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Length::Meter,
            ),
            Energy::Kilocalorie => Energy::Custom(
                Force::Custom(
                    Mass::Custom(base_joules / 4184.0, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Length::Meter,
            ),
            Energy::Erg => Energy::Custom(
                Force::Custom(
                    Mass::Custom(base_joules / 1e-7, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Length::Meter,
            ),
            Energy::TNT => Energy::Custom(
                Force::Custom(
                    Mass::Custom(base_joules / 4.184e9, Box::new(Mass::Kg)),
                    Length::Meter,
                    Time::Second,
                ),
                Length::Meter,
            ),
            Energy::Custom(force, length) => {
                let target_force = Force::Newton.convert_to(force);
                let target_length = Length::Meter.convert_to(length);
                Energy::Custom(target_force, target_length)
            }
        }
    }

    pub fn to_joules(&self) -> f64 {
        match self {
            Energy::Joule => 1.0,
            Energy::Kilojoule => 1000.0,
            Energy::Megajoule => 1_000_000.0,
            Energy::Ev => 1.602176634e-19,
            Energy::KiloEv => 1.602176634e-16,
            Energy::MegaEv => 1.602176634e-13,
            Energy::Calorie => 4.184,
            Energy::Kilocalorie => 4184.0,
            Energy::Erg => 1e-7,
            Energy::TNT => 4.184e9,
            Energy::Custom(force, length) => {
                let newtons = match force {
                    Force::Newton => 1.0,
                    Force::Dyne => 1e-5,
                    Force::Pound => 4.448222,
                    Force::Kg => 9.80665,
                    Force::Custom(mass, len, time) => {
                        let kg = mass.clone().convert_to(Mass::Kg).to_kg();
                        let m = len.clone().convert_to(Length::Meter).to_meters();
                        let s = time.clone().convert_to(Time::Second).to_seconds();
                        kg * (m / (s * s))
                    }
                };
                let meters = length.clone().convert_to(Length::Meter).to_meters();
                newtons * meters
            }
        }
    }

    pub fn to_calories(&self) -> f64 {
        self.to_joules() / 4.184
    }

    pub fn to_kilocalories(&self) -> f64 {
        self.to_joules() / 4184.0
    }

    pub fn to_electron_volts(&self) -> f64 {
        self.to_joules() / 1.602176634e-19
    }

    pub fn to_tnt_equivalent(&self) -> f64 {
        self.to_joules() / 4.184e9
    }
}

#[derive(Debug, Clone)]
pub enum Power {
    Watt,
    Kilowatt,
    Megawatt,
    Hp,
    // Custom power as energy per time
    Custom(Energy, Time),
}

impl Power {
    pub fn convert_to(self, to: Self) -> Self {
        let base_watts = self.to_watts();

        match to {
            Power::Watt => Power::Watt,
            Power::Kilowatt => Power::Custom(
                Energy::Joule,
                Time::Custom(base_watts / 1000.0, Box::new(Time::Second)),
            ),
            Power::Megawatt => Power::Custom(
                Energy::Joule,
                Time::Custom(base_watts / 1_000_000.0, Box::new(Time::Second)),
            ),
            Power::Hp => Power::Custom(
                Energy::Joule,
                Time::Custom(base_watts / 745.7, Box::new(Time::Second)),
            ),
            Power::Custom(energy, time) => {
                let target_energy = Energy::Joule.convert_to(energy);
                let target_time = Time::Second.convert_to(time);
                Power::Custom(target_energy, target_time)
            }
        }
    }

    pub fn to_watts(&self) -> f64 {
        match self {
            Power::Watt => 1.0,
            Power::Kilowatt => 1000.0,
            Power::Megawatt => 1_000_000.0,
            Power::Hp => 745.7,
            Power::Custom(energy, time) => {
                let joules = energy.to_joules();
                let seconds = time.to_seconds();
                joules / seconds
            }
        }
    }

    pub fn to_kilowatts(&self) -> f64 {
        self.to_watts() / 1000.0
    }

    pub fn to_megawatts(&self) -> f64 {
        self.to_watts() / 1_000_000.0
    }

    pub fn to_horsepower(&self) -> f64 {
        self.to_watts() / 745.7
    }
}

#[derive(Debug, Clone)]
pub enum Force {
    Newton,
    Dyne,
    Pound,
    Kg,
    // Custom force as mass times acceleration
    Custom(Mass, Length, Time),
}

impl Force {
    pub fn convert_to(self, to: Self) -> Self {
        let base_newtons = self.to_newtons();

        match to {
            Force::Newton => Force::Newton,
            Force::Dyne => Force::Custom(
                Mass::Custom(base_newtons / 1e-5, Box::new(Mass::Gram)),
                Length::Centimeter,
                Time::Second,
            ),
            Force::Pound => Force::Custom(
                Mass::Custom(base_newtons / 4.448222, Box::new(Mass::Pound)),
                Length::Meter,
                Time::Second,
            ),
            Force::Kg => Force::Custom(
                Mass::Custom(base_newtons / 9.80665, Box::new(Mass::Kg)),
                Length::Meter,
                Time::Second,
            ),
            Force::Custom(mass, length, time) => {
                let target_mass = Mass::Kg.convert_to(mass);
                let target_length = Length::Meter.convert_to(length);
                let target_time = Time::Second.convert_to(time);
                Force::Custom(target_mass, target_length, target_time)
            }
        }
    }

    pub fn to_newtons(&self) -> f64 {
        match self {
            Force::Newton => 1.0,
            Force::Dyne => 1e-5,
            Force::Pound => 4.448222,
            Force::Kg => 9.80665,
            Force::Custom(mass, length, time) => {
                let kg = mass.clone().convert_to(Mass::Kg).to_kg();
                let meters = length.clone().convert_to(Length::Meter).to_meters();
                let seconds = time.clone().convert_to(Time::Second).to_seconds();
                kg * (meters / (seconds * seconds))
            }
        }
    }

    pub fn to_pounds(&self) -> f64 {
        self.to_newtons() / 4.448222
    }

    pub fn to_dynes(&self) -> f64 {
        self.to_newtons() * 100000.0
    }

    pub fn to_kilograms_force(&self) -> f64 {
        self.to_newtons() / 9.80665
    }
}

impl Mass {
    pub fn times_acceleration(&self, length: Length, time: Time) -> Force {
        Force::Custom(self.clone(), length, time)
    }
}

impl Energy {
    pub fn per_time(&self, time: Time) -> Power {
        Power::Custom(self.clone(), time)
    }
}

#[derive(Debug, Clone)]
pub enum ElectricCharge {
    Coulomb,
    ElementaryCharge,
    // Custom charge as current times time
    Custom(Box<Current>, Time),
}

impl ElectricCharge {
    pub fn convert_to(self, to: Self) -> Self {
        let base_coulombs = self.to_coulombs();

        match to {
            ElectricCharge::Coulomb => ElectricCharge::Coulomb,
            ElectricCharge::ElementaryCharge => ElectricCharge::Custom(
                Box::new(Current::Custom(
                    ElectricCharge::Coulomb,
                    Time::Custom(base_coulombs / 1.602176634e-19, Box::new(Time::Second)),
                )),
                Time::Second,
            ),
            ElectricCharge::Custom(current, time) => {
                let target_current = Current::Ampere.convert_to(*current);
                let target_time = Time::Second.convert_to(time);
                ElectricCharge::Custom(Box::new(target_current), target_time)
            }
        }
    }

    pub fn to_coulombs(&self) -> f64 {
        match self {
            ElectricCharge::Coulomb => 1.0,
            ElectricCharge::ElementaryCharge => 1.602176634e-19,
            ElectricCharge::Custom(current, time) => {
                let amperes = match **current {
                    Current::Ampere => 1.0,
                    Current::Milliampere => 0.001,
                    Current::Custom(ref charge, ref t) => {
                        charge
                            .clone()
                            .convert_to(ElectricCharge::Coulomb)
                            .to_coulombs()
                            / t.clone().convert_to(Time::Second).to_seconds()
                    }
                };
                let seconds = time.to_seconds();
                amperes * seconds
            }
        }
    }

    pub fn to_elementary_charges(&self) -> f64 {
        self.to_coulombs() / 1.602176634e-19
    }
}

#[derive(Debug, Clone)]
pub enum Current {
    Ampere,
    Milliampere,
    Custom(ElectricCharge, Time),
}

impl Current {
    pub fn convert_to(self, to: Self) -> Self {
        let base_amperes = self.to_amperes();

        match to {
            Current::Ampere => Current::Ampere,
            Current::Milliampere => Current::Custom(
                ElectricCharge::Coulomb,
                Time::Custom(base_amperes / 0.001, Box::new(Time::Second)),
            ),
            Current::Custom(charge, time) => {
                let target_charge = ElectricCharge::Coulomb.convert_to(charge);
                let target_time = Time::Second.convert_to(time);
                Current::Custom(target_charge, target_time)
            }
        }
    }

    pub fn to_amperes(&self) -> f64 {
        match self {
            Current::Ampere => 1.0,
            Current::Milliampere => 0.001,
            Current::Custom(charge, time) => {
                let coulombs = charge.to_coulombs();
                let seconds = time.to_seconds();
                coulombs / seconds
            }
        }
    }

    pub fn to_milliamperes(&self) -> f64 {
        self.to_amperes() * 1000.0
    }
}

#[derive(Debug, Clone)]
pub enum Frequency {
    Hertz,
    Kilohertz,
    Megahertz,
    Gigahertz,
    // Custom frequency as 1/time
    Custom(Time),
}

impl Frequency {
    pub fn convert_to(self, to: Self) -> Self {
        let base_hertz = self.to_hertz();

        match to {
            Frequency::Hertz => Frequency::Hertz,
            Frequency::Kilohertz => {
                Frequency::Custom(Time::Custom(base_hertz / 1000.0, Box::new(Time::Second)))
            }
            Frequency::Megahertz => Frequency::Custom(Time::Custom(
                base_hertz / 1_000_000.0,
                Box::new(Time::Second),
            )),
            Frequency::Gigahertz => Frequency::Custom(Time::Custom(
                base_hertz / 1_000_000_000.0,
                Box::new(Time::Second),
            )),
            Frequency::Custom(time) => {
                let target_time = Time::Second.convert_to(time);
                Frequency::Custom(target_time)
            }
        }
    }

    pub fn to_hertz(&self) -> f64 {
        match self {
            Frequency::Hertz => 1.0,
            Frequency::Kilohertz => 1000.0,
            Frequency::Megahertz => 1_000_000.0,
            Frequency::Gigahertz => 1_000_000_000.0,
            Frequency::Custom(time) => 1.0 / time.to_seconds(),
        }
    }

    pub fn to_kilohertz(&self) -> f64 {
        self.to_hertz() / 1000.0
    }

    pub fn to_megahertz(&self) -> f64 {
        self.to_hertz() / 1_000_000.0
    }

    pub fn to_gigahertz(&self) -> f64 {
        self.to_hertz() / 1_000_000_000.0
    }
}

#[derive(Debug, Clone)]
pub enum Acceleration {
    MetersPerSecondSquared,
    G, // Earth's gravitational acceleration
    // Custom acceleration
    Custom(Length, Time, Time),
}

impl Acceleration {
    pub fn convert_to(self, to: Self) -> Self {
        // Convert to meters per second squared as base unit
        let base_mps2 = self.to_mps2();

        match to {
            Acceleration::MetersPerSecondSquared => Acceleration::MetersPerSecondSquared,
            Acceleration::G => Acceleration::Custom(
                Length::Custom(base_mps2 / 9.80665, Box::new(Length::Meter)),
                Time::Second,
                Time::Second,
            ),
            Acceleration::Custom(length, time1, time2) => {
                let target_length = Length::Meter.convert_to(length);
                let target_time1 = Time::Second.convert_to(time1);
                let target_time2 = Time::Second.convert_to(time2);
                Acceleration::Custom(target_length, target_time1, target_time2)
            }
        }
    }

    pub fn to_mps2(&self) -> f64 {
        match self {
            Acceleration::MetersPerSecondSquared => 1.0,
            Acceleration::G => 9.80665,
            Acceleration::Custom(length, time1, time2) => {
                let meters = length.clone().convert_to(Length::Meter).to_meters();
                let seconds1 = time1.clone().convert_to(Time::Second).to_seconds();
                let seconds2 = time2.clone().convert_to(Time::Second).to_seconds();
                meters / (seconds1 * seconds2)
            }
        }
    }

    pub fn to_g(&self) -> f64 {
        self.to_mps2() / 9.80665
    }
}

impl Time {
    pub fn frequency(&self) -> Frequency {
        Frequency::Custom(self.clone())
    }
}

impl Length {
    pub fn per_time_squared(&self, time1: Time, time2: Time) -> Acceleration {
        Acceleration::Custom(self.clone(), time1, time2)
    }
}
