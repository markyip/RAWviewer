from typing import Any

MLComputeUnitsCPUOnly: Any
MLComputeUnitsCPUAndGPU: Any
MLComputeUnitsCPUAndNeuralEngine: Any
MLComputeUnitsAll: Any
MLFeatureValue: Any
MLDictionaryFeatureProvider: Any
MLMultiArray: Any
MLMultiArrayDataTypeFloat32: Any
MLMultiArrayDataTypeInt32: Any
MLFeatureTypeMultiArray: Any
MLFeatureTypeImage: Any
MLArrayBatchProvider: Any
MLModelConfiguration: Any
MLModel: Any

def compileModelAtURL_error_(url: Any, error: Any) -> tuple[Any, Any]: ...
def modelWithContentsOfURL_configuration_error_(url: Any, config: Any, error: Any) -> tuple[Any, Any]: ...
def modelWithContentsOfURL_error_(url: Any, error: Any) -> tuple[Any, Any]: ...
