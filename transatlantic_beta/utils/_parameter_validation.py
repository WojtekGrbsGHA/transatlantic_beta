class Constraint:
    def _verify_type(param):
        pass
    def _verify_bounds(param):
        pass
    def _set_parameter_name(self, p_name):
        assert type(p_name) == str
        self.p_name = p_name

    def verify_parameter(self,param):
        self._verify_type(param)
        self._verify_bounds(param)

class Interval(Constraint):
    def __init__(self, interval_type, lower_bound, upper_bound, closed):
        self.interval_type = interval_type
        self.closed = closed

        if interval_type == float:
            self.lower_bound = float(lower_bound)
            self.upper_bound = float(upper_bound)

        elif interval_type == int:
            self.lower_bound = int(lower_bound)
            self.upper_bound = int(upper_bound)

        if closed not in ("left", "right", "neither", "both"):
            raise ValueError("Incorrect interval bounds!")
    
    def _verify_type(self, param):
        try:
            self.interval_type(param)
        except:
            raise ValueError(f"{self.p_name}: {param} cannot be casted into {self.interval_type}")
    def _verify_bounds(self, param):
        t_param = self.interval_type(param)
        if t_param < self.lower_bound or t_param > self.upper_bound:
            raise ValueError(f"Parameter '{self.p_name}': value '{t_param}' not within bounds!")
        if self.closed not in ("left", "both") and t_param == self.lower_bound:
            raise ValueError(f"Parameter '{self.p_name}': value '{t_param}' not within bounds!")
        if self.closed not in ("right", "both") and t_param == self.upper_bound:
            raise ValueError(f"Parameter '{self.p_name}': value '{t_param}' not within bounds!")
        
class Precise(Constraint):
    def __init__(self, type_list, definition_list):
        
        self.type_list = type_list
        self.definition_list = definition_list
    def _verify_type(self, param):
        if type(param) not in self.type_list:
            raise ValueError(f"Parameter {self.p_name}: type [{type(param)}] not supported for this argument")
    def _verify_bounds(self, param):
        if param not in self.definition_list:
            raise ValueError(f"Parameter {self.p_name}: value [{param}] not supported for this argument")

def validate_parameters(constraints_dict: dict[str, Constraint], parameters_dict):
    for parameter in constraints_dict:
        constraint = constraints_dict[parameter]
        constraint._set_parameter_name(parameter)
        constraint.verify_parameter(parameters_dict[parameter])