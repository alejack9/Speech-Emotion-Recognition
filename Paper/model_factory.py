import abc

class ModelFactory(abc.ABC):
  
  def get_model(self, args={}):
    """Should never be called directly."""
    raise NotImplementedError()

