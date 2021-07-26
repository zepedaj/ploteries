from typing import Callable, Union, Dict
from pglib.py import class_name
from dash import Dash
from dash.dependencies import Input, Output, State
import abc
from typing import TYPE_CHECKING


class AbstractInterfaceHook(abc.ABC):

    @property
    @classmethod
    @abc.abstractmethod
    def handler_class(self):
        """
        The handler type stored in the figure_defs table, as a class-level property
        """
        pass

    @classmethod
    def encoded_class_name(cls):
        return class_name(cls).replace('.', '|')

    @abc.abstractmethod
    def build_empty_html(self, figure_handler): pass

    @classmethod
    @abc.abstractmethod
    def create_callbacks(
            cls,
            app: Dash,
            get_hook: Callable[[str], 'AbstractInterfaceHook'],
            callback_args: Dict[str, Union[State, Input, Output]]):
        """
        This method is called from :class:`~.cli_interface.PloteriesLaunchInterface.create_callbacks` to create callbacks for this hook.

        : param app: The Dash object where callbacks are added.
        : param get_hook: Callable that returns an instance of this class. Will be used within callbacks to process requests.
        : param callback_args: Dictionary of callback States, Inputs and Outputs.
        """

    @classmethod
    def _get_id(cls, figure_name, **kwargs):
        return {
            'name': figure_name,
            'type': cls.encoded_class_name(),
            **kwargs
        }
