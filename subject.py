class Subject:
    """
    Base skeleton class for experimental subjects. \\
    This class is meant to be inherited by specific subject implementations (e.g. ideal observer) and defines the interface that all subjects must implement.
    """

    def __init__(self):
        pass

    def before_action(self, o_t):
        """
        This method is called before the subject selects an action. It can be used to process the observation and update internal beliefs.
        """
        pass

    def select_action(self) -> int:
        """
        This method is called to select an action. It should return the selected action as an integer (e.g. 0, 1).
        """
        pass

    def after_action(self, o_t, a_t, r_t):
        """
        This method is called after the subject selects an action and receives a reward. \\
        It can be used to update internal representations based on the emissions at time t: observation o_t, action a_t, reward r_t.
        """
        pass

    @property
    def p_action(self):
        """
        This property should return the subject's inferred probabilities of selecting each action at the current time step, based on its internal representations.
        """
        pass
