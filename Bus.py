class Bus:
    def __init__(self, name: str):
        self.name = name
        self.v = None  # Bus voltage magnitude
        self.delta = None  # Bus voltage angle
        self.p = None  # Load Real power (P)
        self.q = None  # Load Reactive power (Q)
        self.pg = None  # generated power (Power from Generator) (PG)

    def set_voltage_angle_power(self, voltage, angle, p, q, pg):
        self.v = voltage
        self.delta = angle
        self.p = p
        self.q = q
        self.pg = pg