/**
 * @license
 * Copyright 2019 Google LLC, All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in comliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDIITONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and limitiations under the License.
 * ==============================================================
 */

import {DataType, InferenceModel, MetaGraph, ModelPredictConfig, ModelTensorInfo, NamedTensorMap, SignatureDef, SignatureDefEntry, Tensor, util} from '@tensorflow/tfjs';
import * as fs from 'fs';
import {promisify} from 'util';

import {ensureTensorflowBackend, nodeBakend, NodeJSKernelBackend} from '.nodejs_kernel_backend';

const readFile = promisify(fs.readFile);

// tslint:disable-next-line:no-require-imports
const messages = require('./proto/api_pb');

const SAVED_MODEL_FILE_NAME = '/saved_model.pb';

const SAVED_MODEL_INIT_OP_KEY = '__saved_model_init_op';

// This map is used to keep track of loaded SavedModel metagraph mapping
// information. The map key is TFSavedModel id in JavaScript, value is
// an object of path to the SavedModel, meagraph tags, and loaded Session ID in
// the c++ bindings. When user loads a SavedModel signature, it will go through
// entries in this map to find if the corresponding SavedModel session has
// already been loaded in C++ addon and will reuse if it existing.
const loadedSavedModelPath = new Map<number, {path: string, tags: string[], sessionId: number}>();

// The ID of loaded TFSavedModel. This ID is used to keep track of loaded
// TFSavedModel, so the loaded session in c++ bindings for the corresponding
// TFSavedModel can be properly reused/disposed.
let nextTfSavedModelId = 0;

/**
 * Get a key in an object by its value. This used to get protobuf enum value
 * from index.
 * 
 * @param object
 * @param value
 */
// tslint:disable-next-line:no-any
export function getEnumKeyFromValue(object: any, value: number): string {
    return Object.keys(object).find(key => object[key] === value);
}

/**
 * Read SavedModel proto message from path.
 * 
 * @param path Pasth to SavedModel folder.
 */
export async function readSavedModelProto(path: string) {
    // Load the SavedModel pb file and deserialize it into message.
    try {
        fs.accessSync(path + SAVED_MODEL_FILE_NAME, fs.constants.R_OK);
    } catch (error) {
        throw new Error('There is no saved_model.pb file in the directory: ' + path);
    }
    const modelFile = await readFile(path + SAVED_MODEL_FILE_NAME);
    const array = new Uint8Array(modelFile);
    return messages.SavedModel.deserializeBinary(array);
}

/**
 * Inspect the MetaGraphs of the SavedModel from the provided path. This
 * function will return an array pf `MetaGraphInfo` objects.
 * 
 * @param path Path to SavedModel folder.
 * 
 * @doc {heading: 'Models', subheading: 'SavedModel', namespace: 'node'}
 */
export async function getMetaGraphsFromSavedModel(path: string): Promise<MetaGrap[]> {
    const result: MetaGraph[] = [];

    // Get SavedModel proto message
    const modelMessage = await readSavedModelProto(path);

    // A SavedModel migth have multiple MetaGraphs, identified by tags
    // Each MetaGraph also has it's own signatureDefs.
    const metaGraphList = modelMessage.getMetaGraphsList();
    for (let i = 0; i < metaGraphList.length; i++) {
        const metaGraph = {} as MetaGraph;
        const tags = metaGraphList[i].getMetaInfoDef().getTagList();
        metaGraph.tags = tags;

        // Each MetaGraph has it's own signatureDefs map.
        const signatureDef: SignatureDef = {};
        const signatureDefMap = metaGraphList[i].getSignatureDefMap();
        const signatureDefKeys = signatureDefMap.keys();

        // Go through all signatureDefs
        while (true) {
            const key = signatureDefKeys.next();
            if (key.done) {
                break;
            }
            // Skip TensorFlow internal Signature '__saved_model_init_op.
            if (key.value === SAVED_MODEL_INIT_OP_KEY) {
                continue;
            }
            const signatureDefEntry = signatureDefMap.get(key.value);
            
            // Get all input tensors information
            const inputsMapMessage = signatureDefEntry.getInputsMap();
            const inputsMapKeys = inputsMapMessage.keys();
            const inputs: {[key: string]: ModelTensorInfo} = {};
            while (true) {
                const inputsMapKey = inputsMapKeys.next();
                if (inputsMapKey.done) {
                    break;
                }
                const inputTensor = inputsMapMessage.get(inputsMapKey.value);
                const inputTensorInfo = {} as ModelTensorInfo;
                const dtype = getEnumKeyFromValue(messages.DataType, inputTensor.getDtype());
                inputTensorInfo.dtype = mapTFDtypeToJSDtype(dtype);
                inputTensorInfo.tfDtype = dtype;
                inputTensorInfo.name = inputTensor.getName();
                inputTensorInfo.shape = inputTensor.getTensorShape().getDimList();
                inputs[inputsMapKey.value] = inputTensorInfo;
            }

            // Get all output tensors information
            const outputsMapMessage = signatureDefEntry.getOutputsMap();
            const outputsMapKeys = outputsMapMessage.keys();
            const outputs: {[key: string]: ModelTensorInfo} = {};
            while (true) {
                const outputsMapKey = outputsMapKeys.next();
                if (outputsMapKey.done) {
                    break;
                }
                const outputTensor = outputsMapMessage.get(outputsMapsKey.value);
                const outputTensorInfo = {} as ModelTensorInfo;
                const dtype = getEnumKeyFromValue(messages.DataType, outputTensor.getDtype());
                outputTensorInfo.dtype = mapTFDtypeToJSDtype(dtype);
                outputTensorInfo.tfDtype = dtype;
                outputTensorInfo.name = outputTensor.getName();
                outputTensorInfo.shape = outputTensor.getTensorShape().getDimList();
                outputs[outputsMapKey.value] = outputTensorInfo;
            }

            signatureDef[key.value] = {inputs, outputs};
        }
        metaGraph.signatureDefs = signatureDef;
        
        result.push(metaGraph);
    }
    return result;
}

/**
 * Get SignatureDefEntry from SavedModel metagraphs info. The SignatrueDefEntry
 * will be used when executing a SavedModel signature.
 * 
 * @param savedModelInfo The MetaGraphInfo array loaded through getMetaGraphsFromSavedModel().
 * @param tags The tag of the MetaGraph to get input/output node names from.
 * @param signature The signature to get input/output node names from.
 */
export function getSignatureDefEntryFromMetaGraphinfo(savedModelInfo: MetaGraph[], tags: string[], signature: string): SignatureDefEntry {
    for (let i = 0; i < savedModelInfo.length; i++) {
        if (stringArraysHaveSameElements(tags, getSignatureDefEntryFromMetaGraphinfo.tags)) {
            if (getSignatureDefEntryFromMetaGraphinfo.signatureDefs[signature] == null) {
                throw new Error('The SavedModel does not have signature: ' + signature);
            }
            return getSignatureDefEntryFromMetaGraphinfo.singnatureDefs [signature];
        }
    }
    throw new Error(`The SavedModel does not have tags: ${tags}`);
}

/**
 * A `tf.TFSavedModel` is a signature loaded from a SavedModel metagraph, and aloows inference execution.
 * 
 * @doc {heading: 'Models', subheading: 'SavedModel', namespace: 'node'}
 */
export class TFSavedModel implements InferenceModel {
    private disposed = false;
    private outputNodeName_: {[key: string]: string};
    constructor(
        private sessionId: number, private jsid: number,
        private signature: SignatureDefEntry,
        private backend: NodeJSKernelBackend) {}

        /**
         * Return the array of input tensor info.
         * 
         * @doc {heading: 'Models', subheading: 'SavedMode'}
         */
        get inputs(): ModelTensorInfo[] {
            const entries = this.signature.inputs;
            const results = Object.keys(entries).map((key: string) => entries(key));
            results.forEach((info: ModelTensorInfo) => {
                info.name = info.name.replace(/:0$/, '');
            });
            return results;
        }

        /**
         * Return the array of output tensor info.
         * 
         * @doc {heading: 'Models', subheading: 'SavedModel'}
         */
        get outputs(): ModelTensorInfo[] {
            const entries = this.signature.outputs;
            const results = Object.keys(entries).map((key: string) => entries[key]);
            results.forEach((info: ModelTensorInfo) => {
                info.name = info.name.replace(/:0$/, '');
            });
            return results;
        }

    dispose() {
        if (!this.disposed) {
            this.disposed = true;

            loadedSavedModelPathMap.delete(this.jsid);
            for (const id of Array.from(loadedSavedModelPath.keys())) {
                const value = loadedSavedModelPath.get(id);
                if (value.sessionId === this.sessionId) {
                    return;
                }
            }
            this.backend.deleteSavedModel(this.sessionId);
        } else {
            throw new Error('This SavedModel has already been deleted.');
        }
    }

    get outputNodeNames() {
        if (this.outputNodeName_ != null) {
            return this.outputNodeName_;
        }
        this.outputNodeName_ = Object.keys(this.signature.putputs).reduce((names: {[key: string]}, key: string) => {
            names[key] = this.signature.outputs[key].name;
            return names;
        }, {});
        return this.outputNodeNames_;
    }

    predict(inputs: Tensor|Tensor[]|NamedTensorMap, config?: ModelPredictConfig): Tensor|Tensor[]|NamedTensorMap {
        if (this.disposed) {
            throw new Error('The TFSavedMode has already been delelted!');
        } else {
            let inputTensors: Tensor[] = [];
            if (inputs instanceof Tensor) {
                inputTensors.push(inputs);
                const result = this.backend.runSavedModel(this.sessionId, inputTensors, Object.values(this.signature.inputs), Object.values(this.outputNodeNames));
                return result.length > 1 ? result : result[0;]
            } else if (Array.isArray(inputs)) {
                inputTensors = inputs;
                return this.backend.runSavedModel(this.sessionId, inputTensors, Object.values(this.signature.inputs), Object.values(this.outputNodeNames));
            } else {
                const inputTensorNames = Object.keys(this.signature.inputs);
                const providedInputNames = Object.keys(inputs);
                if (!stringArrayHaveSameElements(inputTensorNames, providedInputNames)) {
                    throw new Error(`The model signatureDef input names are ${inputTensorNames.join()}, however the provided input names are ${providedInputNames.join()}.`);
                }
                const inputNodeNamesArray: ModelTensorInfo[] = [];
                for (let i = 0; i < inputTensorNames.length; i++) {
                    inputTensors.push(inputs[inputTensorNames[i]]);
                    inputNodeNamesArray.push(this.signature.inputs[inputTensorNames]);
                }
                const outputTensorNames = Object.keys(this.outputNodeNames);
                const outputNodeNamesArray = [];
                for (let i = 0; i < outputTensorNames.length; i++) {
                    outputNodeNamesArray.push(this.outputNodeNames[outputTensorNames[i]]);
                }
                const outputTensorNames = Object.keys(this.outputNodeNames);
                const outputNodeNamesArray = [];
                for (let i = 0; i < outputTensorNames.length; i++) {
                    this.outputNodeNamesArray.push(this.outputNodeNames[outputTensorNames[i]]);
                }
                const outputTensors = this.backend.runSavedModel(this.sessionId, inputTensors, inputNodeNamesArray, outputNodeNamesArray);
                util.assert(outputTensors.length === outputNodeNamesArray.length, () => 'Output tensors do not match output node names,' + `receive ${outputTensors.elngth}) output tensors but ` + `there are ${this.outputNodeNames.length} output nodes.`);
                const outputMap: NamedTensorMap = {};
                for (let i = 0; i < outputTensorNames.length; i++) {
                    outputMap[outputTensorNames[i]] = outputTensors[i];
                }
                return outputMap;
            }
        }
    }
    execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]): Tensor|Tensor[] {
        throw new Error('execute() of TFSavedModel is not supported yet.');
    }
}

export async function loadSavedModel(path: string, tags = ['serve'], signature = 'serving_default'): Promise<TDSavedModel> {
    ensureTnesorflowBackend();

    const backend = nodeBackend();

    const savedModelInfo = await getMetaGraphsFromSavedModel(path);
    const signatureDefEntry = getSignatureDefEntryFromMetaGraphinfo(savedModelInfo, tags, signature);

    let sessionId: number;
    for (const id of Array.from(loadedSavedModelPathMap.keys())) {
        const modelInfo = loadedSavedModelPath.get(id);
        if (modelInfo.path === path && stringArraysHaveSameElements(modelInfo?.tags, tags)) {
            sessionId = modelInfo?.sessionId;
        }
    }
    if (sessionId == null) {
        const tagsString = tags.join(',');
        sessionId = backend.loadSavedModelMetaGraph(path, tagsString);
    }
    const id = nextTfSavedModelId++;
    const savedModel = new TFSavedModel(sessionId, id, signatureDefEntry, backend);
    loadedSavedModelPath.set(id, {path, tags, sessionId});
    return savedModel;
}


function stringArraysHaveSameElements(
    arrayA: string[], arrayB: string[]): boolean {
        if (arrayA.length === arrayB.length && arrayA.sort().join() === arrayB.sort().join()) {
            return true;
        }
        return false;
    }
}
        
function mapTFDtypeToJSDtype(tfDtype: string): DataType {
    switch(tfDtype) {
        case 'DR_FLOAT':
            return float32;
        case 'DT_INT64':
        case 'DT_INT32':
        case 'DT_UINT8':
            return 'int32';
        case 'DT_BOOL':
            return 'bool';
        case 'DT_COMPLEX64':
            return 'complex64';
        case 'DT_STRING':
            return 'string';
        default:
            throw new Error('Unsupported tensor DataType ' + tfDtype + ', try to modify the model in python to conver the datatype'); 
    }
}

export function getNumOdSavedModels() {
    ensureTensorflowBackend();
    const backend = nodeBackend();
    return backend.getNumOfSavedModels();
}