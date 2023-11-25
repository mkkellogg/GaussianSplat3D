import * as THREE from 'three';
import { Viewer } from './Viewer.js';

export class RenderableViewer extends THREE.Group {

    constructor(options = {}) {
        super();

        options.selfDrivenMode = false;
        options.useBuiltInControls = false;
        options.rootElement = null;
        options.ignoreDevicePixelRatio = false;
        options.initializeFromExternalUpdate = true;

        this.viewer = new Viewer(options);

        this.callbackMesh = this.createCallbackMesh();
        this.add(this.callbackMesh);
        this.callbackMesh.onBeforeRender = this.onBeforeRender.bind(this);

    }

    addSceneFromFile(fileURL, options = {}) {
        if (options.showLoadingSpinner !== false) options.showLoadingSpinner = true;
        return this.viewer.loadFile(fileURL, options).then(() => {
            this.add(this.viewer.splatMesh);
        });
    }

    addScenesFromFiles(fileURLs, splatBufferOptions, meshOptions, showLoadingSpinner) {
        if (showLoadingSpinner !== false) showLoadingSpinner = true;
        return this.viewer.loadFiles(fileURLs, splatBufferOptions, meshOptions, showLoadingSpinner).then(() => {
            this.add(this.viewer.splatMesh);
        });
    }

    onBeforeRender(renderer, scene, camera) {
        this.viewer.update(renderer, scene, camera);
    }

    createCallbackMesh() {
        const geometry = new THREE.SphereGeometry(1, 8, 8);
        const material = new THREE.MeshBasicMaterial();
        material.colorWrite = false;
        material.depthWrite = false;
        const mesh = new THREE.Mesh(geometry, material);
        mesh.frustumCulled = false;
        return mesh;
    }

}