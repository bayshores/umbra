"""Isolated analytic source-binding fixtures, never actual street/model evidence."""
import copy
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/evaluation'))
import adapt_nvidia as adapter
import matched


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='analytic-binding-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        raw = self.root/'raw';raw.mkdir()
        self.shape = [322, 738]
        for role, value in [('left', 17), ('right', 211)]:
            Image.fromarray(np.full((*self.shape, 3), value, dtype=np.uint8)).save(raw/(role+'.png'))
        Image.fromarray(np.full(self.shape, 2560, dtype=np.uint16)).save(raw/'depth.png')
        (raw/'calibration.txt').write_text('S_rect_101: 738 322\nS_rect_103: 738 322\nP_rect_101: 10 0 3 0 0 10 2 0 0 0 1 0\nP_rect_103: 10 0 1 -12 0 10 2 0 0 0 1 0\n')
        prepare = adapter.common.load_script('prepare-stereo-crop.py')
        prepare.prepare(SimpleNamespace(left=raw/'left.png', right=raw/'right.png', depth=raw/'depth.png',
            calibration=raw/'calibration.txt', output=self.root/'crop', crop=[1,1,736,320],
            depth_divisor=256, expected_disparity=False))
        def record(name, **fields):
            p=raw/name
            return dict(path=str(p), bytes=p.stat().st_size, sha256=matched.digest(p.read_bytes()), readOnlyExternal=True, **fields)
        processing=dict(originalDimensions=self.shape, inputDimensions=self.shape, outputDimensions=self.shape,
            crop=None,pad=None,resize=None,normalization='analytic fixture only',
            coordinateConvention='OpenCV-left-rectified-x-right-y-down-z-forward',depthConvention='axialZ')
        self.observation=dict(id='analytic-fixture-NO-MODEL-EXECUTION',split='development',siteGroup='analytic-only',
            physicalSite=None, sources=[record('left.png',role='left'),record('right.png',role='right')],
            calibration=dict(artifact=record('calibration.txt')),admission=dict(status='unresolved',reasons=['analytic fixture']),
            reference=dict(raster=record('depth.png',encoding='uint16png',quantity='axialDepthMeters',shape=self.shape,divisor=256),
                processing=processing,independent=False,method='analytic fixture',timing='not measured',uncertainty='not applicable',limits=['analytic fixture only']))
        self.crop_path=self.root/'crop/manifest.json'

    def bind(self):
        return adapter.bind_crop(self.observation,self.root,self.crop_path)

    def repin_crop(self, name, data):
        path=self.crop_path.parent/name;path.write_bytes(data)
        crop=adapter.common.parse(self.crop_path.read_bytes())
        crop['outputs'][name]=dict(sha256=matched.digest(data),bytes=len(data))
        self.crop_path.write_bytes(adapter.common.canonical(crop))

    def test_labeled_positive_binding_derives_original_matrices(self):
        bound=self.bind()
        self.assertEqual(bound['shape'],self.shape)
        self.assertEqual(bound['crop'],[1,1,736,320])
        self.assertEqual(set(bound['sourceBytes']),{'left','right'})

    def test_repinned_pixel_change_rejected(self):
        with Image.open(self.crop_path.parent/'left.png') as image:
            data=np.array(image)
        data[0,0,0]+=1
        buffer=io.BytesIO();Image.fromarray(data).save(buffer,format='PNG')
        self.repin_crop('left.png',buffer.getvalue())
        with self.assertRaisesRegex(ValueError,'crop pixels'):
            self.bind()

    def test_repinned_coherent_crop_calibration_change_rejected(self):
        path=self.crop_path.parent/'calibration.json';cal=adapter.common.parse(path.read_bytes())
        # Still internally coherent T@P, but no longer matches original calibration.
        cal['original']['left_projection'][0][0]=11
        cal['transformed']['left_projection'][0][0]=11
        self.repin_crop('calibration.json',adapter.common.canonical(cal))
        with self.assertRaisesRegex(ValueError,'original projection'):
            self.bind()

    def test_swapped_original_sources_rejected(self):
        self.observation['sources'][0]['role']='right';self.observation['sources'][1]['role']='left'
        with self.assertRaisesRegex(ValueError,'source identities'):
            self.bind()

    def test_original_byte_change_rejected(self):
        Path(self.observation['sources'][0]['path']).write_bytes(b'changed analytic fixture')
        with self.assertRaisesRegex(ValueError,'bytes/hash mismatch'):
            self.bind()

    def test_wrong_grid_rejected(self):
        self.observation['reference']['raster']['shape']=[321,738]
        processing=self.observation['reference']['processing']
        for key in ('originalDimensions','inputDimensions','outputDimensions'):
            processing[key]=[321,738]
        with self.assertRaisesRegex(ValueError,'original dimensions'):
            self.bind()

    def test_refused_split_role_units(self):
        original=copy.deepcopy(self.observation)
        for change in ('split','role','units'):
            with self.subTest(change=change):
                self.observation=copy.deepcopy(original)
                if change=='split':self.observation['split']='heldoutCandidate'
                if change=='role':self.observation['sources'][0]['role']='image'
                if change=='units':self.observation['reference']['raster']['divisor']=128
                with self.assertRaises(ValueError):self.bind()

    def test_analytic_publisher_retains_producer_processing(self):
        # Mocked validation is explicitly isolated fixture evidence, no model call.
        crop=self.bind();producer_dir=self.root/'analytic-producer';producer_dir.mkdir()
        disparity=np.full((320,736),4,dtype='<f4').tobytes();validity=np.ones((320,736),dtype='u1').tobytes()
        (producer_dir/'disparity.float32le').write_bytes(disparity);(producer_dir/'validity.uint8').write_bytes(validity)
        producer=dict(schemaVersion=1,observationId=self.observation['id'],model=dict(provider='analytic-fixture',id='NO-MODEL',version='test'),
            sources={'cropManifest':adapter.common.record(self.crop_path,crop['cropBytes'],producer_dir),
                **{name:adapter.common.record(self.crop_path.parent/name,data,producer_dir) for name,data in crop['files'].items()}},
            processing=dict(originalDimensions=self.shape,inputDimensions=[320,736],outputDimensions=[320,736],
                crop={'x':1,'y':1,'width':736,'height':320},disparityConvention=adapter.disparity_registration.CONVENTION),
            prediction=adapter.pin('disparity.float32le',disparity,shape=[320,736],encoding='float32le',quantity='disparityPixels'),
            validity=adapter.pin('validity.uint8',validity,shape=[320,736],encoding='uint8'),
            execution=dict(status='succeeded',kind='analytic fixture NOT model execution'))
        op=self.root/'observations.json';op.write_bytes(adapter.common.canonical(dict(schemaVersion=1,observations=[self.observation])))
        pp=self.root/'plan.json';pp.write_bytes(b'{}')
        manifest=producer_dir/'prediction.json';manifest.write_bytes(adapter.common.canonical(producer))
        output=self.root/'registered'
        args=(op,matched.digest(op.read_bytes()),pp,matched.digest(pp.read_bytes()),producer_dir,matched.digest(manifest.read_bytes()),self.root/'NO-MODEL',output)
        with patch.object(adapter.nvidia_stereo,'cached',return_value=producer):
            result=adapter.adapt(*args)
        saved=adapter.common.parse((output/'prediction.json').read_bytes())
        self.assertEqual(saved['derivation']['originalInferenceProcessing'],producer['processing'])
        self.assertEqual(saved['processing']['outputDimensions'],self.shape)
        self.assertEqual(result['counts']['validReference'],322*738)
        self.assertEqual(result['counts']['matchedPrediction'],320*736)
        self.assertEqual(result['counts']['invalidPredictionOnReference'],322*738-320*736)
        self.assertEqual(saved['execution']['usage'],dict(newInferenceRuns=0,cloudRequests=0))
        inventory={p.name:p.read_bytes() for p in output.iterdir()}
        with self.assertRaisesRegex(ValueError,'Existing output'):
            adapter.adapt(*args)
        self.assertEqual(inventory,{p.name:p.read_bytes() for p in output.iterdir()})

    def test_absent_prediction_never_publishes_success(self):
        op=self.root/'observations.json';op.write_bytes(b'{}');pp=self.root/'plan.json';pp.write_bytes(b'{}')
        output=self.root/'absent-result'
        with self.assertRaises(FileNotFoundError):
            adapter.adapt(op,matched.digest(op.read_bytes()),pp,matched.digest(pp.read_bytes()),self.root/'absent','0'*64,self.root/'NO-MODEL',output)
        self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
