import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastmcp import Client

from facade_mcp.cli import main
from facade_mcp import cli
from facade_mcp.image_server import server
from test_image_inspection import FakeExperts, execute, row, setup


def test_image_mcp_tool_registered():
    async def check():
        async with Client(server) as client:
            tools = await client.list_tools()
            by_name = {tool.name:tool for tool in tools}
            assert {'facade_image_inspect','facade_image_failure_summary','facade_image_record_feedback',
                    'facade_image_rules_latest','facade_image_rule_propose','facade_image_rule_validate',
                    'facade_image_rule_cycle','facade_image_validate_vision'} == set(by_name)
            schema = by_name['facade_image_inspect'].input_schema
            assert set(schema['required']) == {'request', 'model_config_path'}
    asyncio.run(check())


def test_image_client_export_uses_copy_without_exporting_secret(tmp_path, monkeypatch, capsys):
    output = tmp_path/'mcp.local.json'
    monkeypatch.setenv('FACADE_IMAGE_API_KEY','fixture-secret-must-not-export')
    monkeypatch.setattr(sys,'argv',['facade-image','--workspace',str(tmp_path/'workspace'),
                                   'mcp-config','--include-image','--output',str(output)])
    main()
    config = json.loads(output.read_text(encoding='utf-8'))
    image = config['mcpServers']['image']
    assert image['args'] == ['-m','facade_mcp.image_server']
    assert str(Path(cli.__file__).resolve().parents[1]) in image['env']['PYTHONPATH']
    assert 'fixture-secret-must-not-export' not in output.read_text(encoding='utf-8')
    capsys.readouterr()


@pytest.mark.parametrize('fault',['hash','dimensions','coordinate_space','bbox_convention'])
def test_expert_cannot_mix_sources_or_coordinate_conventions(tmp_path,fault):
    request,store,folder,_,_ = setup(tmp_path)
    class BadExpert(FakeExperts):
        async def call(self,name,arguments):
            result = await super().call(name,arguments)
            if fault == 'hash':
                result['images'][0]['source_sha256'] = '0'*64
            elif fault == 'dimensions':
                result['images'][0]['width'] += 1
            elif fault == 'coordinate_space':
                result['coordinate_space'] = 'normalized'
            else:
                result['bbox_convention'] = 'xywh'
            return result
    with pytest.raises(ValueError):
        execute(request,store,folder,BadExpert(full=[row()]))
