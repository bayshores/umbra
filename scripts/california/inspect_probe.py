#!/usr/bin/env python3
"""Inspect an explicit AV2 probe offline; acquire no data and create no geometry.

A matching city tag and inventory never certify a crossing, independent physical
site, reference accuracy, camera rectification or holdout eligibility.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET

NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def pin(path, data):
    return {'path': str(path.resolve()), 'external': True, 'readOnly': True,
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def inspect(plan_path, listing_paths, local_root=None):
    plan_data = plan_path.read_bytes()
    plan = json.loads(plan_data)
    log_id = plan['log_id']
    if not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', log_id):
        raise ValueError('Expected an AV2 UUID log id')
    if plan['split'] not in ('train', 'val', 'test'):
        raise ValueError('Unknown publisher split')
    prefix = f"datasets/av2/sensor/{plan['split']}/{log_id}/"
    for field in ('stereo_timestamp_ns', 'lidar_timestamp_ns'):
        if not isinstance(plan[field], str) or not re.fullmatch(r'[0-9]+', plan[field]):
            raise ValueError('Nanosecond timestamps must be decimal strings')
    camera_ns, lidar_ns = int(plan['stereo_timestamp_ns']), int(plan['lidar_timestamp_ns'])
    files = plan['files']
    names = [item['path'] for item in files]
    for name in names:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or '..' in path.parts or chr(92) in name:
            raise ValueError('Artifact path must be relative without traversal')
    if not 1 <= len(files) <= 20 or len(set(names)) != len(names):
        raise ValueError('Selection must contain 1-20 unique artifacts')
    sizes = [item['listing_bytes'] for item in files]
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError('Artifact byte counts must be positive integers')
    if sum(sizes) > 16_000_000 or sum(sizes) != plan['total_listing_bytes']:
        raise ValueError('Selection total changed or exceeds the 16MB inspection bound')
    required = {'annotations.feather', 'city_SE3_egovehicle.feather',
                'calibration/egovehicle_SE3_sensor.feather', 'calibration/intrinsics.feather',
                f'sensors/cameras/stereo_front_left/{camera_ns}.jpg',
                f'sensors/cameras/stereo_front_right/{camera_ns}.jpg',
                f'sensors/lidar/{lidar_ns}.feather'}
    maps = [name for name in names if name.startswith(f'map/log_map_archive_{log_id}____') and name.endswith('.json')]
    if not required.issubset(names) or len(maps) != 1:
        raise ValueError('Selection lacks exact stereo, LiDAR, pose, calibration, annotations or map')
    sources = [pin(plan_path, plan_data)]
    objects = {}
    for path in listing_paths:
        data = path.read_bytes()
        root = ET.fromstring(data)
        listed_prefix = root.findtext('s3:Prefix', namespaces=NS, default='')
        allowed_prefixes = {prefix + suffix for suffix in (
            '', 'calibration/', 'map/', 'sensors/cameras/stereo_front_left/',
            'sensors/cameras/stereo_front_right/', 'sensors/lidar/')}
        if root.findtext('s3:Name', namespaces=NS) != 'argoverse' or listed_prefix not in allowed_prefixes:
            raise ValueError('Listing bucket or exact prefix does not match the selected probe')
        if root.findtext('s3:IsTruncated', namespaces=NS, default='true') != 'false':
            raise ValueError('Incomplete listing cannot verify the inventory')
        sources.append(pin(path, data))
        for entry in root.findall('s3:Contents', NS):
            key = entry.findtext('s3:Key', namespaces=NS)
            size = int(entry.findtext('s3:Size', namespaces=NS))
            if not key.startswith(listed_prefix):
                raise ValueError('Object outside selected log')
            if key in objects and objects[key] != size:
                raise ValueError('Conflicting listing byte counts')
            objects[key] = size
    inventory, failures = [], []
    approved = plan.get('approval_received') is True
    if approved and local_root is None:
        failures.append('Approved payload inspection requires explicit --local-root')
    for item in files:
        name = item['path']
        observed = objects.get(prefix + name)
        record = {'path': name, 'expectedBytes': item['listing_bytes'], 'listedBytes': observed,
                  'listingMatches': observed == item['listing_bytes'], 'payloadStatus': 'notRead'}
        if not record['listingMatches']:
            failures.append('Object missing or listing size changed: ' + name)
        if local_root is not None and approved:
            root_path = local_root.resolve()
            file_path = local_root / name
            if not file_path.resolve().is_relative_to(root_path):
                raise ValueError('Local artifact escapes the explicit input root')
            if not file_path.is_file():
                record['payloadStatus'] = 'missing'
                failures.append('Approved required payload missing: ' + name)
            else:
                if any(part.is_symlink() for part in (file_path, *file_path.parents)):
                    raise ValueError('Payload symlinks are not accepted')
                with file_path.open('rb') as stream:
                    data = stream.read(item['listing_bytes'] + 1)
                record['payloadStatus'] = 'sizeMatches' if len(data) == item['listing_bytes'] else 'sizeMismatch'
                # A size-mismatched bounded prefix is not an artifact byte pin.
                if record['payloadStatus'] == 'sizeMatches':
                    sources.append(pin(file_path, data))
                else:
                    failures.append('Local payload size changed: ' + name)
        inventory.append(record)
    match = re.search(r'____([A-Z]{3})_city_', maps[0])
    city_code = match.group(1) if match else None
    california_city = city_code == 'PAO' and objects.get(prefix + maps[0]) == next(item['listing_bytes'] for item in files if item['path'] == maps[0])
    reasons = failures + ['Calibration, metric scale and rectification not validated',
                          'Crossing visibility and usable camera support unverified',
                          'Distinct physical sites and untouched holdout unverified',
                          'Reference uncertainty, dynamic timing and model exposure unverified']
    if not approved:
        reasons.insert(0, 'AV2 terms approval pending; no payloads read')
    if not california_city:
        reasons.insert(0, 'California city candidate not established by publisher metadata')
    if any(item['payloadStatus'] != 'sizeMatches' for item in inventory):
        reasons.append('Complete selected payload inventory not verified')
    observation = {
        'id': f'av2:{log_id}:{camera_ns}', 'siteGroup': None, 'split': 'unassigned',
        'geographyEvidence': {'cityCode': city_code, 'californiaCityCandidate': california_city,
                              'basis': 'Publisher S3 map filename only', 'physicalSiteVerified': False},
        'sources': sources,
        'license': {'sourceUrl': plan['terms_url'], 'version': 'Publisher terms; recheck before acquisition',
                    'acceptanceState': 'planRecordsLocalApproval' if approved else 'pending',
                    'redistributionLimits': 'Subject to publisher terms and separately verified use/redistribution permissions'},
        'calibration': {'artifact': 'calibration/egovehicle_SE3_sensor.feather', 'status': 'uninspected',
                        'conventions': 'Publisher egovehicle_SE3_sensor and undistorted pinhole images; verify SDK'},
        'reference': {'measurementMethod': 'Candidate raw LiDAR projected at camera time through publisher poses',
                      'dependency': 'Independent modality for stereo-only predictions; shares pose/calibration; not independent of LiDAR reconstruction',
                      'timing': {'stereoTimestampNs': str(camera_ns), 'lidarTimestampNs': str(lidar_ns),
                                 'absoluteDeltaNs': str(abs(lidar_ns - camera_ns))},
                      'uncertaintyLimits': 'Unknown calibration/point uncertainty and dynamic motion; sparse returns are not whole-segment visibility labels'},
        'admission': {'status': 'rejected' if failures else 'unresolved', 'reasons': reasons}}
    return {'schemaVersion': 1, 'observations': [observation],
            'inspection': {'scope': 'Offline selected inventory; no geometry, acquisition or release admission',
                           'listingBytes': sum(sizes), 'inventory': inventory, 'failures': failures,
                           'verifiedUsableCaliforniaSites': 0, 'verifiedHeldoutSites': 0,
                           'code': pin(Path(__file__), Path(__file__).read_bytes())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--listing', type=Path, action='append', required=True)
    parser.add_argument('--local-root', type=Path, help='Explicit read-only log root; payloads ignored while approval is pending')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    protected = [args.plan, *args.listing, Path(__file__)]
    if output in [path.resolve() for path in protected] or (
            args.local_root is not None and output.is_relative_to(args.local_root.resolve())):
        raise ValueError('Output cannot overwrite an inspection input or write inside the raw input root')
    report = inspect(args.plan, args.listing, args.local_root)
    with args.output.open('x', encoding='utf-8') as output_file:
        output_file.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'admission': report['observations'][0]['admission']['status'],
                      'failures': len(report['inspection']['failures']), 'usableCaliforniaSites': 0}))
    return 1 if report['inspection']['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
