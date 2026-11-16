"""Run report JavaScript, rather than only checking template substrings."""

import json
import shutil
import subprocess

import pytest

from tests.paths import REPO_ROOT

TEMPLATE = REPO_ROOT / "src/gsMap/report/static/template.html"


@pytest.fixture
def run_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise report JavaScript")
    text = TEMPLATE.read_text()
    app_code = text[text.index("function gsMapApp()") :].split("</script>", 1)[0]
    setup = """
        const plot = {replaceChildren() {this.cleared = true;}};
        global.window = {};
        global.document = {getElementById: id => id === 'cauchyHeatmap' ? plot : null};
        global.Plotly = {
            react(div, traces, layout) {
                div.traces = traces; div.layout = layout;
                div.data = traces;
                div.renderCount = (div.renderCount || 0) + 1;
                return Promise.resolve();
            },
            purge(div) {div.purged = true; delete div.data;}
        };
        const app = gsMapApp();
        app.getPlotlyConfig = () => ({});
        app.$nextTick = fn => fn();
    """

    def run(body):
        result = subprocess.run(
            [node, "--input-type=module"],
            input=app_code + setup + body,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    return run


def test_volume_keeps_nonspatial_panels_but_disables_2d(run_js):
    result = run_js("""
        app.reportMeta={dataset_type:'spatial3D', spatial_layout:'volume'};
        console.log(JSON.stringify(app.currentSections));
    """)
    assert result["show3d"] is True
    assert result["show2dDistribution"] is False
    for key in ("showUMAP", "showManhattan", "showEnrichment", "showGSS"):
        assert result[key] is True


def test_3d_fullscreen_resizes_on_enter_and_external_exit_without_resetting(run_js):
    result = run_js("""
        let requests=0, resizes=0, focused=0, hidden=0;
        const tooltip={parentElement:null,style:{display:'block'}};
        const body={appendChild(node){node.parentElement=this;}};
        const panel={appendChild:body.appendChild,requestFullscreen:async()=>{requests++;}};
        tooltip.parentElement=body;document.body=body;
        document.getElementById=id=>({'sec-spatial3d':panel,gsmapPointHoverTooltip:tooltip,
            spatial3dFullscreenButton:{focus:()=>focused++}}[id] || null);
        app.resizeSpatial3dViews=async()=>{resizes++;};
        app.hidePointHoverTooltip=()=>{hidden++;tooltip.style.display='none';};
        app.renderSpatial3dViews=()=>{throw Error('Fullscreen must not rebuild traces');};
        app.spatial3dPointSize=.9;app.spatial3dOpacity=.5;
        const pending=app.toggleSpatial3dFullscreen();
        const immediate=requests;await pending;
        document.fullscreenElement=panel;await app.onSpatial3dFullscreenChange();
        const entered={active:app.spatial3dFullscreen,parent:tooltip.parentElement===panel};
        await app.onSpatial3dFullscreenChange(); // Duplicate event must not resize again.
        document.fullscreenElement=null;await app.onSpatial3dFullscreenChange(); // Browser Esc.
        console.log(JSON.stringify({immediate,entered,exited:!app.spatial3dFullscreen,
            parent:tooltip.parentElement===body,resizes,focused,hidden,
            size:app.spatial3dPointSize,opacity:app.spatial3dOpacity,busy:app.spatial3dFullscreenBusy}));
    """)
    assert result == {
        "immediate": 1,
        "entered": {"active": True, "parent": True},
        "exited": True,
        "parent": True,
        "resizes": 2,
        "focused": 2,
        "hidden": 2,
        "size": 0.9,
        "opacity": 0.5,
        "busy": False,
    }


def test_3d_fullscreen_exit_rejection_and_unsupported_browser_are_safe(run_js):
    result = run_js("""
        let exits=0;
        const panel={requestFullscreen:async()=>{throw Error('Permission denied');}};
        document.getElementById=id=>id==='sec-spatial3d' ? panel:null;
        document.exitFullscreen=async()=>{exits++;document.fullscreenElement=null;};
        document.fullscreenElement=panel;
        await app.toggleSpatial3dFullscreen();
        await app.toggleSpatial3dFullscreen();
        const rejected=app.spatial3dFullscreenNotice;
        delete panel.requestFullscreen;
        await app.toggleSpatial3dFullscreen();
        console.log(JSON.stringify({exits,rejected,unsupported:app.spatial3dFullscreenNotice,
            active:app.spatial3dFullscreen,busy:app.spatial3dFullscreenBusy,error:app.spatial3dError}));
    """)
    assert result["exits"] == 1
    assert "permissions" in result["rejected"]
    assert "not available" in result["unsupported"]
    assert not result["active"]
    assert not result["busy"]
    assert not result["error"]


def test_2d_fullscreen_uses_2d_state_and_leaves_3d_untouched(run_js):
    result = run_js("""
        const panel={requestFullscreen:async()=>{document.fullscreenElement=panel;}};
        document.getElementById=id=>id==='sec-gsmap' ? panel:null;
        document.exitFullscreen=async()=>{document.fullscreenElement=null;};
        let resizes=0;
        app.resizeSpatial2dViews=async()=>{resizes++;};
        app.resizeSpatial3dViews=()=>{throw Error('Must not resize 3D');};
        app.renderSpatial2DPlot=()=>{throw Error('Must not rebuild traces');};
        app.spatial2dPointScale=2;app.spatial2dCompareKind='samples';
        await app.toggleSpatial2dFullscreen();await app.onSpatial2dFullscreenChange();
        const entered=app.spatial2dFullscreen;
        await app.onSpatial3dFullscreenChange();
        await app.toggleSpatial2dFullscreen();await app.onSpatial2dFullscreenChange();
        console.log(JSON.stringify({entered,resizes,exited:!app.spatial2dFullscreen,
            untouched:!app.spatial3dFullscreen,scale:app.spatial2dPointScale,kind:app.spatial2dCompareKind,
            busy:app.spatial2dFullscreenBusy}));
    """)
    assert result == {
        "entered": True,
        "resizes": 2,
        "exited": True,
        "untouched": True,
        "scale": 2,
        "kind": "samples",
        "busy": False,
    }


@pytest.mark.parametrize("fail", [False, True])
def test_2d_fullscreen_resize_preserves_independent_viewports_and_flags(run_js, fail):
    result = run_js(
        """
        const fail=FAIL;
        const a={data:[{x:[1]}],__gsMapApplyingViewport:false},b={data:[{x:[2]}]};
        document.getElementById=id=>({spatial2dPlotTrait:a,spatial2dPlotAnnotation:b}[id]);
        app.waitForLayoutPaint=async()=>{};
        let guarded=true,resizes=0,sizes=0;
        Plotly.Plots={resize:async p=>{
            resizes++;guarded=guarded && a.__gsMapApplyingViewport && b.__gsMapApplyingViewport;
            if(fail)throw Error('Resize failed');
        }};
        app.updateSpatial2dPointSize=p=>{sizes++;};
        let failed=false;
        try {await app.resizeSpatial2dViews();} catch(e){failed=true;}
        console.log(JSON.stringify({resizes,sizes,guarded,failed,
            restored:a.__gsMapApplyingViewport===false && b.__gsMapApplyingViewport===undefined,
            coordinates:[a.data[0].x[0],b.data[0].x[0]]}));
    """.replace("FAIL", str(fail).lower())
    )
    assert result == {
        "resizes": 2,
        "sizes": 0 if fail else 2,
        "guarded": True,
        "failed": fail,
        "restored": True,
        "coordinates": [1, 2],
    }


def test_point_tooltip_is_created_inside_fullscreen_top_layer(run_js):
    result = run_js("""
        const nodes={};
        document.createElement=()=>({setAttribute(){}});
        const body={appendChild(node){nodes[node.id]=node;node.parentElement=this;}};
        const panel={appendChild:body.appendChild};
        document.body=body;document.getElementById=id=>nodes[id] || null;
        document.fullscreenElement=panel;
        const first=app.getPointHoverTooltip(),inside=first.parentElement===panel;
        document.fullscreenElement=null;
        const second=app.getPointHoverTooltip();
        console.log(JSON.stringify({inside,same:first===second,restored:second.parentElement===body,
            id:first.id,style:first.className}));
    """)
    assert result == {
        "inside": True,
        "same": True,
        "restored": True,
        "id": "gsmapPointHoverTooltip",
        "style": "gsmap-hover-tooltip",
    }


def test_annotation_overview_density_and_focus_are_reversible(run_js):
    result = run_js("""
        window.GSMAP_ANNOTATION_SCHEMAS={celltype:{categories:['01','2','NA'],colors:['red','blue','green']}};
        app.selectedAnnotationCategory='celltype'; app.selectedTrait='IQ';
        app.renderAnnotationCategoryViews=()=>{};
        app.cauchyData=['01','2','NA'].map(annotation=>({trait:'IQ', annotation_name:'celltype', type:'aggregated', annotation}));
        app.toggleAnnotationCategory('01');
        const focused={categories:app.selectedAnnotationCategories,table:app.filteredCauchyTable.map(r=>r.annotation),
            visible: ['01','2','NA'].filter(c=>app.isAnnotationCategoryVisible(c))};
        app.toggleAnnotationCategory('01');
        console.log(JSON.stringify({focused,restored:!app.hasAnnotationSelection,
            table:app.filteredCauchyTable.length, zero:app.overviewScore(0),missing:app.overviewScore(null),
            flat:app.overviewRidgePath([0,0]),ridge:app.overviewRidgePath([1,0,0])}));
    """)
    assert result["focused"] == {"categories": ["01"], "table": ["01"], "visible": ["01"]}
    assert result["restored"]
    assert result["table"] == 3
    assert result["zero"] == "0.00"
    assert result["missing"] == "—"
    assert result["flat"] == ""
    assert result["ridge"].endswith("Z")
    assert "NaN" not in result["ridge"]


def test_sidebar_multiselection_filters_all_linked_views_and_clear_restores(run_js):
    result = run_js("""
        window.GSMAP_ANNOTATION_SCHEMAS={celltype:{categories:['01','2','NA'],colors:['red','blue','green']}};
        app.selectedAnnotationCategory='celltype'; app.selectedTrait='IQ';
        app.renderAnnotationCategoryViews=()=>{};
        app.cauchyData=['01','2','NA'].map(annotation=>({trait:'IQ', annotation_name:'celltype', type:'aggregated', annotation,mlog10_p_cauchy:1}));
        app.cauchyHeatmapData=app.cauchyData;
        app.umapData={celltype:['01','2','NA','01']};
        const schema={categories:['01','2','NA'],codes:new Uint8Array([0,1,2,0])};
        const x=new Float32Array([0,1,2,3]);
        app.toggleAnnotationCategory('01'); app.toggleAnnotationCategory('2');
        const chosen=[...app.selectedAnnotationCategories];
        const table=app.filteredCauchyTable.map(r=>r.annotation).sort();
        await app.renderCauchyHeatmap();const heatmap=plot.traces[0].y.slice().sort();
        const points=Array.from(app.filterCategoricalPlotPoints(x,x,null,schema,'celltype').indices);
        const umap=app.visibleUmapPointIndices();
        app.toggleAnnotationCategory('unknown');
        const unchanged=app.selectedAnnotationCategories.slice();
        app.toggleAnnotationCategory('01'); const removed=app.selectedAnnotationCategories.slice();
        app.selectedSample='other'; app.overviewScope='sample';
        const preserved=app.selectedAnnotationCategories.slice();
        app.overviewQuery='01'; app.overviewLimit=160;
        app.clearAnnotationSelection();
        console.log(JSON.stringify({chosen,table,heatmap,points,umap,unchanged,removed,preserved,
            restored:app.filteredCauchyTable.length,allPoints:app.filterCategoricalPlotPoints(x,x,null,schema,'celltype').indices,
            allUmap:app.visibleUmapPointIndices(),active:app.hasAnnotationSelection,
            query:app.overviewQuery,limit:app.overviewLimit}));
    """)
    assert result == {
        "chosen": ["01", "2"],
        "table": ["01", "2"],
        "heatmap": ["01", "2"],
        "points": [0, 1, 3],
        "umap": [0, 1, 3],
        "unchanged": ["01", "2"],
        "removed": ["2"],
        "preserved": ["2"],
        "restored": 3,
        "allPoints": None,
        "allUmap": None,
        "active": False,
        "query": "",
        "limit": 80,
    }


def test_sidebar_all_explicitly_selected_can_still_remove_one(run_js):
    result = run_js("""
        window.GSMAP_ANNOTATION_SCHEMAS={domain:{categories:['1','2'],colors:[]}};
        app.selectedAnnotationCategory='domain'; app.renderAnnotationCategoryViews=()=>{};
        app.toggleAnnotationCategory(1);app.toggleAnnotationCategory('2');
        const all=app.selectedAnnotationCategories.slice();
        app.toggleAnnotationCategory('1');
        console.log(JSON.stringify({all,selected:app.selectedAnnotationCategories,
            visible:['1','2'].filter(c=>app.isAnnotationCategoryVisible(c)),
            otherAnnotation:app.isAnnotationCategoryVisible('unrelated','other')}));
    """)
    assert result == {
        "all": ["1", "2"],
        "selected": ["2"],
        "visible": ["2"],
        "otherAnnotation": True,
    }


def test_annotation_overview_ignores_stale_scope_loads(run_js):
    result = run_js("""
        app.reportMeta={association_overview_version:1};app.selectedAnnotationCategory='celltype';
        app.overviewIndex={traits:{IQ:{celltype:{all:{key:'iq'}}},AF:{celltype:{all:{key:'af'}}}}};
        const pending={};app.loadOverviewShard=e=>new Promise(resolve=>pending[e.key]=resolve);
        app.selectedTrait='IQ';const old=app.loadAssociationOverview();
        app.selectedTrait='AF';const latest=app.loadAssociationOverview();
        pending.af({trait:'AF',annotation:'celltype',sample:null,rows:[]});await latest;
        pending.iq({trait:'IQ',annotation:'celltype',sample:null,rows:[]});await old;
        console.log(JSON.stringify({trait:app.overviewData.trait,loading:app.overviewLoading,error:app.overviewError}));
    """)
    assert result == {"trait": "AF", "loading": False, "error": ""}


def test_spatial_maps_start_in_compare_without_2d_signal_fading(run_js):
    result = run_js("""
        console.log(JSON.stringify({
            mode2d: app.spatial2dViewMode, mode3d: app.spatial3dViewMode,
            emphasis2d: app.spatial2dAssociationOpacityStrength,
            emphasis3d: app.spatial3dAssociationOpacityStrength,
            opacity: app.associationOpacityArray(
                new Float32Array([0, 1, 5, 10]), 0, 10, 0.9,
                app.spatial2dAssociationOpacityStrength, 0),
            camera: app.spatial3dLayout().scene.camera
        }));
    """)
    assert result["mode2d"] == result["mode3d"] == "compare"
    assert result["emphasis2d"] == 0
    assert result["emphasis3d"] == 0.65
    assert result["opacity"] == 0.9
    assert result["camera"]["center"] == {"x": 0, "y": 0, "z": 0}
    assert result["camera"]["eye"] == pytest.approx({"x": 1.8, "y": 1.8, "z": 1.8})


def test_report_top_gene_default_keeps_complete_correlation_list(run_js):
    result = run_js("""
        const initial = app.topCorrGenes;
        app.loadGenePccData = async () => Array.from({length: 60}, (_, i) =>
            ({gene: 'gene' + i, PCC: 1 - i / 100}));
        app.selectedTrait = 'IQ';
        await app.updateGeneList('IQ');
        const top10 = app.topGenesList.filter(g => g.isTop).length;
        const full10 = app.topGenesList.length;
        app.topCorrGenes = 50;
        await app.updateGeneList('IQ');
        console.log(JSON.stringify({initial, top10, full10,
            top50: app.topGenesList.filter(g => g.isTop).length, full50: app.topGenesList.length}));
    """)
    assert result == {"initial": 10, "top10": 10, "full10": 60, "top50": 50, "full50": 60}


def test_spatial2d_hover_index_matches_exact_nearest_point_and_rebuilds(run_js):
    result = run_js("""
        let seed = 7123;
        const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2**32);
        const x = Float32Array.from({length: 20000}, () => rand()*120-10);
        const y = Float32Array.from({length: 20000}, () => rand()*120-10);
        x[0]=NaN; y[1]=Infinity; x[2]=x[3]=50; y[2]=y[3]=50;
        const axis = (range, length, reverse=false) => ({range, _length:length,
            l2p(v) {return (reverse ? 1-(v-this.range[0])/(this.range[1]-this.range[0]) :
                (v-this.range[0])/(this.range[1]-this.range[0])) * this._length;}});
        const hoverPlot = {data:[{x,y}], _fullLayout:{xaxis:axis([0,100],800), yaxis:axis([100,0],600,true)}};
        const index=app.spatial2dHoverIndex(hoverPlot);
        const reused=app.spatial2dHoverIndex(hoverPlot)===index;
        let mismatches=0;
        for (let q=0;q<100;q++) {
            const px=rand()*800, py=rand()*600;
            let best=-1, distance=25;
            for(let i=0;i<x.length;i++) {
                const xx=Math.fround(hoverPlot._fullLayout.xaxis.l2p(x[i]));
                const yy=Math.fround(hoverPlot._fullLayout.yaxis.l2p(y[i]));
                if(xx<0||xx>800||yy<0||yy>600) continue;
                const d=(xx-px)**2+(yy-py)**2;
                if(d<distance||(d===distance&&i>best)) {best=i;distance=d;}
            }
            if(app.findSpatial2dHoverPoint(index,px,py)!==best) mismatches++;
        }
        const tie=app.findSpatial2dHoverPoint(index,400,300);
        hoverPlot._fullLayout.xaxis.range=[20,80];
        const zoom=app.spatial2dHoverIndex(hoverPlot);
        hoverPlot.data[0]={x:new Float32Array([50]),y:new Float32Array([50])};
        const filtered=app.spatial2dHoverIndex(hoverPlot);
        console.log(JSON.stringify({mismatches,reused,tie,zoomChanged:zoom!==index,
            filteredChanged:filtered!==zoom,n:filtered.next.length,
            outside:app.findSpatial2dHoverPoint(filtered,-1,10),
            center:app.findSpatial2dHoverPoint(filtered,400,300)}));
    """)
    assert result == {
        "mismatches": 0,
        "reused": True,
        "tie": 3,
        "zoomChanged": True,
        "filteredChanged": True,
        "n": 1,
        "outside": -1,
        "center": 0,
    }


def test_spatial3d_projection_depth_clipping_and_camera_cache(run_js):
    result = run_js("""
        const identity=()=>Float64Array.from([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
        const params={model:identity(),view:identity(),projection:identity()};
        const gl={cameraParams:params,canvas:{getBoundingClientRect:()=>({width:100,height:100})},
            bounds:[[-1,-1,-1],[1,1,1]]};
        const target={__gsMap3dPoints:{x:[0,0,4,NaN],y:[0,0,0,0],z:[.5,-.5,0,0],
            visible:Uint32Array.from([0,1,2,3]),size:6,opacity:1},
            _fullLayout:{scene:{_scene:{glplot:gl,dataScale:[1,1,1]}}}};
        const index=app.spatial3dHoverIndex(target);
        const reused=app.spatial3dHoverIndex(target)===index;
        const nearest=app.findSpatial3dHoverPoint(index,50,50);
        params.view[12]=.5;
        const moved=app.spatial3dHoverIndex(target);
        const hit=app.findSpatial3dHoverPoint(moved,75,50);
        gl.objects=[];
        const empty=app.spatial3dHoverIndex(target)===null;
        gl.objects=[{}];gl.contextLost=true;
        const lost=app.spatial3dHoverIndex(target)===null;
        console.log(JSON.stringify({reused,nearest,index:moved.point[hit],cameraChanged:moved!==index,empty,lost,
            oldPosition:app.findSpatial3dHoverPoint(moved,50,50),outside:app.findSpatial3dHoverPoint(moved,200,50)}));
    """)
    assert result == {
        "reused": True,
        "nearest": 1,
        "index": 1,
        "cameraChanged": True,
        "oldPosition": -1,
        "outside": -1,
        "empty": True,
        "lost": True,
    }


def test_spatial3d_hover_resolves_off_center_occlusion_and_rotation(run_js):
    result = run_js("""
        const identity=()=>Float64Array.from([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
        const params={model:identity(),view:identity(),projection:identity()};
        const gl={cameraParams:params,canvas:{getBoundingClientRect:()=>({width:100,height:100})},
            bounds:[[-1,-1,-1],[1,1,1]]};
        const target={__gsMap3dPoints:{x:[0,.04],y:[0,0],z:[.5,-.5],visible:Uint32Array.from([0,1]),
            size:2,opacity:1},
            _fullLayout:{scene:{_scene:{glplot:gl,dataScale:[1,1,1]}}}};
        const before=app.spatial3dHoverIndex(target);
        const hidden=app.findSpatial2dHoverPoint(before,50,50);
        const visible=app.findSpatial3dHoverPoint(before,50,50);
        params.view[0]=-1;params.view[10]=-1;
        const rotated=app.spatial3dHoverIndex(target);
        console.log(JSON.stringify({hidden,visible,rotated:app.findSpatial3dHoverPoint(rotated,48,50),
            reused:app.spatial3dHoverIndex(target)===rotated}));
    """)
    assert result == {"hidden": 0, "visible": 1, "rotated": 0, "reused": True}


def test_spatial3d_hover_footprints_visibility_and_size_cache(run_js):
    result = run_js("""
        const identity=()=>Float64Array.from([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
        const gl={cameraParams:{model:identity(),view:identity(),projection:identity()},
            canvas:{getBoundingClientRect:()=>({width:100,height:100})},bounds:[[-1,-1,-1],[1,1,1]]};
        // Row 1 is nearer (z=-.5) but one pixel to the right of row 0.
        const points={x:[0,.04],y:[0,0],z:[.5,-.5],visible:Uint32Array.from([0,1]),size:.4,opacity:1};
        const target={__gsMap3dPoints:points,_fullLayout:{scene:{_scene:{glplot:gl,dataScale:[1,1,1]}}}};
        const small=app.spatial3dHoverIndex(target);
        const throughGap=app.findSpatial3dHoverPoint(small,50,50);
        target.__gsMap3dPoints={...points,size:2};
        const big=app.spatial3dHoverIndex(target);
        const occluded=app.findSpatial3dHoverPoint(big,50,50);
        const proximity=app.findSpatial3dHoverPoint(big,45.5,50);
        const invisible=[];
        for(const update of [{visible:Uint32Array.from([0])},{opacity:0},{size:0}]) {
            target.__gsMap3dPoints={...points,size:2,...update};
            invisible.push(app.findSpatial3dHoverPoint(app.spatial3dHoverIndex(target),50,50));
        }
        console.log(JSON.stringify({throughGap,occluded,proximity,invisible,
            rebuilt:big!==small,empty:app.findSpatial3dHoverPoint(big,10,10)}));
    """)
    assert result == {
        "throughGap": 0,
        "occluded": 1,
        "proximity": 1,
        # A row filtered out is never picked; transparent or zero-size sets pick nothing.
        "invisible": [0, -1, -1],
        "rebuilt": True,
        "empty": -1,
    }


def test_spatial3d_transparent_order_is_global_and_camera_dependent(run_js):
    result = run_js("""
        const identity=()=>Float64Array.from([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
        // Interleave rows as if they came from different signal-opacity bins.
        const xyz=new Float32Array([0,0,-1, 0,0,1, 0,0,-2, 0,0,.5]);
        const camera={view:identity(),model:identity()};
        let sorted=app.spatial3dDepthOrder(xyz,camera);
        const front=Array.from(sorted.order);
        camera.view[14]=100;
        const reused=app.spatial3dDepthOrder(xyz,camera,sorted)===sorted;
        camera.view[10]=-1;
        sorted=app.spatial3dDepthOrder(xyz,camera,sorted);
        console.log(JSON.stringify({front,reused,back:Array.from(sorted.order)}));
    """)
    assert result == {"front": [2, 0, 3, 1], "reused": True, "back": [1, 3, 0, 2]}


def test_spatial3d_transparent_order_matches_full_model_view_transform(run_js):
    result = run_js("""
        const model=Float64Array.from([2,0,0,0,0,3,0,0,0,0,.1,0,-.5,3,1,1]);
        const view=Float64Array.from([.8,0,.6,0,0,1,0,0,-.6,0,.8,0,1,2,3,1]);
        const xyz=Float32Array.from({length:3000},(_,i)=>Math.sin(i*2.7));
        const transformed=Array.from({length:1000},(_,i)=>{
            const x=xyz[3*i]*2-.5,y=xyz[3*i+1]*3+3,z=xyz[3*i+2]*.1+1;
            return {i,z:view[2]*x+view[6]*y+view[10]*z+view[14]};
        }).sort((a,b)=>a.z-b.z||a.i-b.i).map(p=>p.i);
        const actual=app.spatial3dDepthOrder(xyz,{model,view});
        console.log(JSON.stringify({equal:transformed.every((i,k)=>i===actual.order[k])}));
    """)
    assert result["equal"]


def test_spatial3d_transparent_order_reallocates_for_new_geometry(run_js):
    result = run_js("""
        const identity=()=>Float64Array.from([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
        const camera={view:identity(),model:identity()};
        let order=app.spatial3dDepthOrder(new Float32Array([0,0,-1]),camera);
        const xyz=new Float32Array([0,0,1,0,0,-1]);
        order=app.spatial3dDepthOrder(xyz,camera,order);
        console.log(JSON.stringify({rows:Array.from(order.order),positions:order.positions===xyz}));
    """)
    assert result == {"rows": [1, 0], "positions": True}


def test_spatial3d_projection_rebuilds_missing_renderer_traces(run_js):
    result = run_js("""
        const full={scene:'scene',visible:true,_length:2,marker:{size:2}};
        const native={pickSlots:1};native._trace={data:full,scatterPlot:native};
        let rebuilt=0,created=0,redrawn=0,disposed=0,updates=0;
        const renderer={objects:[],add(o){this.objects.push(o)},remove(o){this.objects.splice(this.objects.indexOf(o),1)},redraw(){redrawn++}};
        const scene={glplot:renderer,plot(data){if(data[0]!==full)throw Error('Wrong data');rebuilt++;renderer.objects.push(native)}};
        const target={_fullData:[full],_fullLayout:{scene:{_scene:scene}},layout:{}};
        app.createSpatial3dPointLayer=(r,s)=>{created++;let last=null;return {__gsMapPointLayer:true,
            update(points){updates++;const changed=points!==last;last=points;return changed;},dispose(){disposed++}}};
        target.__gsMap3dPoints={id:1};
        app.prepareSpatial3dRenderer(target);
        const slots=native.pickSlots;
        app.prepareSpatial3dRenderer(target);
        target.__gsMap3dPoints={id:2};app.prepareSpatial3dRenderer(target);
        const layer=renderer.__gsMapPointLayer;
        target.__gsMap3dPoints=null;app.prepareSpatial3dRenderer(target);
        console.log(JSON.stringify({rebuilt,created,redrawn,disposed,updates,slots,
            removed:!renderer.objects.includes(layer)&&!renderer.__gsMapPointLayer,
            suppressed:!native.isOpaque()&&!native.isTransparent(),traceLinked:layer._trace===native._trace}));
    """)
    assert result == {
        "rebuilt": 1,
        "created": 1,
        "redrawn": 3,
        "disposed": 1,
        "updates": 3,
        "slots": 0,
        "removed": True,
        "suppressed": True,
        "traceLinked": True,
    }


def test_spatial3d_hover_preserves_thin_section_depth(run_js):
    result = run_js("""
        const identity=()=>Float64Array.from([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
        const gl={cameraParams:{model:identity(),view:identity(),projection:identity()},
            canvas:{getBoundingClientRect:()=>({width:100,height:100})},bounds:[[-1,-1,-1],[1,1,1]]};
        const target={__gsMap3dPoints:{x:[.02,0],y:[0,0],z:[-.995000002,-.995000001],
            visible:Uint32Array.from([0,1]),size:2,opacity:1},
            _fullLayout:{scene:{_scene:{glplot:gl,dataScale:[1,1,1]}}}};
        const index=app.spatial3dHoverIndex(target);
        console.log(JSON.stringify({picked:app.findSpatial3dHoverPoint(index,50,50),
            different:index.depth[0]!==index.depth[1]}));
    """)
    assert result == {"picked": 0, "different": True}


def test_spatial3d_fully_opaque_overrides_signal_fading_without_changing_colors(run_js):
    result = run_js("""
        const values=Float32Array.from([0,5,10,2.5,NaN,7]);
        const visible=Uint32Array.from([0,1,2,3,4]);
        const rgb=(out,i)=>Array.from(out.rgba.slice(i*4,i*4+3),v=>Math.round(v*255));
        const opaque=app.spatial3dTraitColors(values,visible,['#000000','#ffffff'],0,10,1,8,0,.65);
        app.spatial3dOpacity=1;
        const label=app.spatial3dAssociationOpacityLabel;
        const faded=app.spatial3dTraitColors(values,visible,['#000000','#ffffff'],0,10,.8,8,0,.65);
        const alpha=out=>Array.from(visible,i=>Number(out.rgba[i*4+3].toFixed(4)));
        console.log(JSON.stringify({opaque:opaque.opaque,alphas:alpha(opaque),
            colors:Array.from(visible,i=>rgb(opaque,i)),sameColors:visible.every(i=>rgb(opaque,i).join()===rgb(faded,i).join()),
            label,fadedOpaque:faded.opaque,faded:alpha(faded).some(a=>a<.8),max:Math.max(...alpha(faded)),
            hiddenRowUntouched:opaque.rgba[5*4+3]===0}));
    """)
    assert result["opaque"] is True
    assert result["alphas"] == [1, 1, 1, 1, 1]
    # Plotly's colorscale: linear RGB, clamped, rounded; non-numeric values use #444.
    assert result["colors"] == [
        [0, 0, 0],
        [128, 128, 128],
        [255, 255, 255],
        [64, 64, 64],
        [68, 68, 68],
    ]
    assert result["sameColors"]
    assert result["label"] == "Off · opaque"
    assert result["fadedOpaque"] is False
    assert result["faded"]
    assert result["max"] <= 0.8
    assert result["hiddenRowUntouched"]


def test_spatial3d_opacity_bins_preserve_hover_source_indices(run_js):
    result = run_js("""
        // Bin means are the opacity of the former per-bin traces.
        const values=Float32Array.from([0,5,10,9.9]); // 10 and 9.9 share the top bin
        const visible=Uint32Array.from([0,1,2,3]);
        const out=app.spatial3dTraitColors(values,visible,['#000000','#ffffff'],0,10,.8,10,0,.65);
        const opacity=Array.from(values,v=>app.associationOpacityForValue(v,0,10,.8,.65,0));
        const top=(opacity[2]+opacity[3])/2;
        app.selectedAnnotationCategory='celltype';
        const annotation={categories:['A','B'],colors:['red','blue'],codes:[0,1,1,0,0]};
        const points={x:[1,2,3,4,5],y:[1,2,3,4,5],z:[1,2,3,4,5]};
        const key=app.setSpatial3dHoverContexts({id:'trait'},points,annotation,['red','blue'],
            {cmin:0,cmax:10,values:[0,5,10,9.5,1]});
        const context=app._pointHoverContexts[key];
        console.log(JSON.stringify({sharedBin:Math.abs(out.rgba[2*4+3]-top)<1e-6&&Math.abs(out.rgba[3*4+3]-top)<1e-6,
            ownBin:Math.abs(out.rgba[3]-opacity[0])<1e-6,
            source:app.pointHoverSourceIndex({pointNumber:4},context),
            value:app.pointMarkerColorValue({data:{marker:{color:points.hoverValues}},pointNumber:4}),
            kind:context.hoverKind,key:points.hoverContextKey===key}));
    """)
    assert result == {
        "sharedBin": True,
        "ownBin": True,
        "source": 4,
        "value": 1,
        "kind": "trait",
        "key": True,
    }


def test_spatial3d_visible_rows_follow_categories_cutoff_and_finite_coordinates(run_js):
    result = run_js("""
        app.selectedAnnotationCategory='celltype';
        const annotation={categories:['A','B'],colors:['red','blue'],codes:Uint16Array.from([0,1,1,0,1,0])};
        const values=Float32Array.from([1,5,NaN,8,9,0.5]);
        const x=Float32Array.from([0,1,2,3,NaN,5]),y=new Float32Array(6),z=new Float32Array(6);
        const all=Array.from(app.spatial3dVisibleIndices(6,annotation,'celltype',values,0,x,y,z));
        const cutoff=Array.from(app.spatial3dVisibleIndices(6,annotation,'celltype',values,2,x,y,z));
        app.selectedAnnotationCategories=['B'];
        const onlyB=Array.from(app.spatial3dVisibleIndices(6,annotation,'celltype',values,0,x,y,z));
        const bounds=app.spatial3dVisibleBounds(x,y,z,Uint32Array.from([1,3]));
        const traces=app.spatial3dProxyTraces(bounds,{size:1,cmin:0,cmax:10,showscale:true},'IQ');
        console.log(JSON.stringify({all,cutoff,onlyB,bounds,points:traces[0].x.length,color:traces[0].marker.color,
            skip:traces[0].hoverinfo,empty:app.spatial3dProxyTraces(null,{},'IQ').length}));
    """)
    assert result == {
        "all": [0, 1, 2, 3, 5],
        "cutoff": [1, 3],
        "onlyB": [1, 2],
        "bounds": [[1, 0, 0], [3, 0, 0]],
        "points": 2,
        "color": [0, 0],
        "skip": "skip",
        "empty": 0,
    }


def test_spatial3d_category_colors_match_discrete_scale(run_js):
    result = run_js("""
        const out=app.spatial3dCategoryColors(Uint16Array.from([0,1,2,7]),Uint32Array.from([0,1,3]),
            ['#ff0000','#00ff00','#0000ff'],.5);
        const rgb=i=>Array.from(out.rgba.slice(i*4,i*4+4),v=>Number((v*255).toFixed(1)));
        console.log(JSON.stringify({rows:[rgb(0),rgb(1),rgb(3)],hidden:rgb(2),opaque:out.opaque}));
    """)
    assert result == {
        "rows": [[255, 0, 0, 127.5], [0, 255, 0, 127.5], [0, 0, 255, 127.5]],
        "hidden": [0, 0, 0, 0],
        "opaque": False,
    }


def test_spatial3d_renders_tiny_proxy_traces_not_point_arrays(run_js):
    result = run_js("""
        const n=5000, plots={};
        for(const id of ['spatial3dPlotTrait','spatial3dPlotAnnotation']) plots[id]={id,on(){},addEventListener(){}};
        document.getElementById=id=>plots[id]||null;
        app.plotlyReady=true;app.bindSpatial3dHover=()=>{};app.bindSpatial3dCameraSync=()=>{};
        const x=Float32Array.from({length:n},(_,i)=>i%97),y=Float32Array.from({length:n},(_,i)=>i%13),
            z=Float32Array.from({length:n},(_,i)=>i%7);
        app.spatial3dBase={x,y,z,positions:new Float32Array(n*3)};
        app.spatial3dTraitData={values:Float32Array.from({length:n},(_,i)=>i%11),cmin:0,cmax:10,q95:9};
        app.selectedAnnotationCategory='celltype';
        app.spatial3dAnnotationData={categories:['A','B'],colors:['#ff0000','#0000ff'],codes:Uint16Array.from({length:n},(_,i)=>i%2)};
        app.spatial3dAnnotationCache={celltype:app.spatial3dAnnotationData};
        app.spatial3dViewMode='compare';
        await app.renderSpatial3dViews();
        const trait=plots.spatial3dPlotTrait, annotation=plots.spatial3dPlotAnnotation;
        console.log(JSON.stringify({traitPoints:trait.traces[0].x.length,annotationPoints:annotation.traces[0].x.length,
            traitRows:trait.__gsMap3dPoints.visible.length,sharedRows:trait.__gsMap3dPoints.visible===annotation.__gsMap3dPoints.visible,
            colorbar:trait.traces[0].marker.showscale,range:[trait.traces[0].x,trait.traces[0].z]}));
    """)
    assert result == {
        "traitPoints": 2,
        "annotationPoints": 2,
        "traitRows": 5000,
        "sharedRows": True,
        "colorbar": True,
        "range": [[0, 96], [0, 6]],
    }


def test_spatial3d_synced_plots_share_depth_orders(run_js):
    result = run_js("""
        const identity=()=>Float64Array.from([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
        const positions=Float32Array.from({length:3000},(_,i)=>Math.sin(i*1.7));
        const visible=Uint32Array.from({length:1000},(_,i)=>i);
        const leading={positions,visible},trailing={positions,visible};
        const first={view:identity(),model:identity()},second={view:identity(),model:identity()};
        second.view[2]=.3;
        const a1=app.spatial3dSharedDepthOrder(leading,first,[1,1,1]);
        const a2=app.spatial3dSharedDepthOrder(leading,second,[1,1,1]);
        const b1=app.spatial3dSharedDepthOrder(trailing,first,[1,1,1]);
        const b2=app.spatial3dSharedDepthOrder(trailing,second,[1,1,1]);
        console.log(JSON.stringify({reusedLagging:b1===a1,reusedCurrent:b2===a2,distinct:a1!==a2,
            versions:a1.version!==a2.version,cache:positions.__gsMapDepthOrders.length}));
    """)
    assert result == {
        "reusedLagging": True,
        "reusedCurrent": True,
        "distinct": True,
        "versions": True,
        "cache": 2,
    }


def test_spatial2d_hover_cancels_pending_frames_and_ignores_plotly_events(run_js):
    result = run_js("""
        const frames=new Map(); let serial=0, shown=0, hidden=0;
        global.requestAnimationFrame=fn=>{frames.set(++serial,fn);return serial;};
        global.cancelAnimationFrame=id=>frames.delete(id);
        const flush=()=>{const f=[...frames.values()]; frames.clear();f.forEach(fn=>fn());};
        const events={}, dom={}, win={};
        window.addEventListener=(name,fn)=>win[name]=fn;
        const trace={x:[50],y:[50],meta:{hoverContextKey:'key'}};
        const axis={range:[0,100],_length:100,_offset:0,l2p:v=>v};
        const hoverPlot={data:[trace],_fullLayout:{xaxis:axis,yaxis:axis},
            getClientRects:()=>[{}],getBoundingClientRect:()=>({left:0,top:0}),
            addEventListener:(name,fn)=>dom[name]=fn,on:(name,fn)=>events[name]=fn};
        app._pointHoverContexts.key={};
        app.showPointHoverTooltip=()=>shown++;
        app.hidePointHoverTooltip=()=>hidden++;
        app.bindSpatial2dHover(hoverPlot); app.bindSpatial2dHover(hoverPlot);
        const p={clientX:50.2,clientY:50.5,buttons:0};
        dom.pointermove(p);dom.pointermove(p);flush();
        const once=shown;
        dom.pointermove(p);dom.pointerleave();flush();
        dom.pointermove(p);dom.pointerdown();flush();
        dom.pointermove(p);win.scroll();flush();
        dom.pointermove(p);events.plotly_beforeplot();flush();
        app.loadingSampleData=true;dom.pointermove(p);flush();app.loadingSampleData=false;
        dom.pointermove({...p,buttons:1});flush();
        const canceled=shown;
        dom.pointermove(p);flush();
        win.blur();
        console.log(JSON.stringify({once,canceled,shown,hidden,pending:frames.size,
            nativeHoverBound:!!events.plotly_hover,owner:app._pointHoverOwner}));
    """)
    assert result["once"] == result["canceled"] == 1
    assert result["shown"] == result["hidden"] == 2
    assert result["pending"] == 0
    assert result["nativeHoverBound"] is False
    assert result["owner"] is None


def test_hover_cleanup_rejects_stale_events_and_respects_panel_owner(run_js):
    result = run_js("""
        const blur = [];
        window.addEventListener = (name, handler) => {if(name === 'blur') blur.push(handler);};
        let shown = 0, hidden = 0;
        app.showPointHoverTooltip = () => shown++;
        app.hidePointHoverTooltip = () => hidden++;
        const makePlot = () => ({dom: {}, events: {}, registrations: 0,
            on(name, fn) {this.events[name] = fn;},
            addEventListener(name, fn) {this.dom[name] = fn; this.registrations++;}});
        const a = makePlot(), b = makePlot();
        app.bindPointHoverTooltip(a);
        app.bindPointHoverTooltip(a);
        app.bindPointHoverTooltip(b);
        const key = 'spatial2d\\u001ftrait';
        app._pointHoverContexts[key] = {};
        const pointer = {clientX: 20, clientY: 30, timeStamp: 100, buttons: 0};
        const event = {event: pointer, points: [{data: {meta: {hoverContextKey: key}}}]};
        const counts = [];
        a.dom.pointerenter(pointer); a.events.plotly_hover(event); counts.push(shown);
        a.dom.pointerleave(); a.events.plotly_hover(event); counts.push(shown);
        a.dom.pointermove({...pointer, buttons: 1}); a.events.plotly_hover(event); counts.push(shown);
        a.dom.pointermove(pointer); a.dom.pointerdown(); a.events.plotly_hover(event); counts.push(shown);
        a.dom.pointermove(pointer);
        a.__gsMapHoverNotBefore = 101; a.events.plotly_hover(event); counts.push(shown);
        a.__gsMapHoverNotBefore = 0;
        app.loadingSpatial2dTraitData = true; a.events.plotly_hover(event); counts.push(shown);
        app.loadingSpatial2dTraitData = false;
        a.dom.pointermove({...pointer, clientX: 23}); a.events.plotly_hover(event); counts.push(shown);
        a.dom.pointermove(pointer);
        delete app._pointHoverContexts[key]; a.events.plotly_hover(event); counts.push(shown);
        app._pointHoverContexts[key] = {};
        a.dom.pointermove({...pointer, clientX: 20.75, clientY: 30.5});
        a.events.plotly_hover(event); counts.push(shown);
        b.dom.pointerenter(pointer); b.events.plotly_hover(event); counts.push(shown);
        const before = hidden;
        a.events.plotly_unhover();
        const lateUnhoverIgnored = hidden === before;
        blur.forEach(fn => fn()); b.events.plotly_hover(event); counts.push(shown);
        b.dom.pointerenter(pointer); b.events.plotly_hover(event);
        b.dom.pointercancel(); b.events.plotly_hover(event); counts.push(shown);
        console.log(JSON.stringify({counts, hidden, lateUnhoverIgnored, registrations:a.registrations}));
    """)
    assert result["counts"] == [1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 4]
    assert result["hidden"] >= 3
    assert result["lateUnhoverIgnored"]
    assert result["registrations"] == 6


def test_spatial2d_markers_follow_actual_viewport_zoom_and_manual_scale(run_js):
    result = run_js("""
        const data = {point_size: 6, point_geometry: {diameter: 1.4}, coordinate_ranges: [100, 200]};
        const target = {_fullLayout: {
            xaxis: {range: [0, 100], _length: 400}, yaxis: {range: [200, 0], _length: 800}}};
        const initial = app.resolveSpatial2dPointSize(data, target);
        target._fullLayout.xaxis.range = [0, 50]; target._fullLayout.yaxis.range = [100, 0];
        const zoomed = app.resolveSpatial2dPointSize(data, target);
        app.spatial2dPointScale = 2;
        const manual = app.resolveSpatial2dPointSize(data, target);
        target._fullLayout.xaxis._length /= 2; target._fullLayout.yaxis._length /= 2;
        const smaller = app.resolveSpatial2dPointSize(data, target);
        console.log(JSON.stringify({initial, zoomed, manual, smaller}));
    """)
    assert result == pytest.approx(
        {"initial": 5.6, "zoomed": 11.2, "manual": 22.4, "smaller": 11.2}
    )


def test_spatial2d_point_resize_is_bounded_and_only_changes_marker_size(run_js):
    result = run_js("""
        let callback, binds = 0; const changes = [];
        const data = {point_size: 6, point_geometry: {diameter: 20}};
        app.currentSampleData = data;
        const p = {on(name, cb) {binds++; callback = cb;},
            _fullLayout: {xaxis: {range:[0,1], _length:100}, yaxis: {range:[0,1], _length:100}},
            data: [{marker: {size:6, color:[1,2], cmin:0, cmax:10, opacity:0.9}}]};
        Plotly.restyle = (p, change) => {changes.push(change); p.data[0].marker.size = change['marker.size'];};
        app.bindSpatial2dPointSizing(p, data, 'trait');
        app.bindSpatial2dPointSizing(p, data, 'trait');
        callback(); callback();
        app.currentSampleData = {};
        callback();
        console.log(JSON.stringify({binds, changes, marker:p.data[0].marker, label:app.spatial2dPointSizePixels}));
    """)
    assert result["binds"] == 1
    assert result["changes"] == [{"marker.size": 96}]
    assert result["marker"] == {"size": 96, "color": [1, 2], "cmin": 0, "cmax": 10, "opacity": 0.9}
    assert result["label"] == 96


def test_spatial2d_point_sizing_before_first_layout_and_without_geometry(run_js):
    result = run_js("""
        const p = {clientWidth: 474, clientHeight: 820};
        const first = app.resolveSpatial2dPointSize({point_geometry: {diameter:1.4}, point_size:6,
            coordinate_ranges:[100,200]}, p);
        const old = app.resolveSpatial2dPointSize({point_size:6}, p);
        console.log(JSON.stringify({first, old}));
    """)
    assert result == pytest.approx({"first": 5.04, "old": 6})


def test_spatial2d_autorange_refinement_is_bounded(run_js):
    result = run_js("""
        let calls = 0;
        const data = {point_size:6, point_geometry:{diameter:2}};
        app.currentSampleData = data;
        const p = {data:[{marker:{size:6}}], _fullLayout:{
            xaxis:{range:[0,100],_length:400},yaxis:{range:[0,100],_length:400}}};
        app.bindSpatial2dPointSizing(p, data, 'trait');
        Plotly.restyle = (plot, change) => {
            calls++; plot.data[0].marker.size=change['marker.size'];
            plot._fullLayout.xaxis._length *= 0.8;
            plot._fullLayout.yaxis._length *= 0.8;
        };
        app.updateSpatial2dPointSize(p);
        await new Promise(resolve => setTimeout(resolve, 0));
        console.log(JSON.stringify({calls}));
    """)
    assert result["calls"] == 3


def test_umap_resize_only_when_container_dimensions_differ(run_js):
    result = run_js("""
        let calls = 0;
        Plotly.Plots = {resize(p) {
            calls++;
            p._fullLayout.width = p.clientWidth;
            p._fullLayout.height = p.clientHeight;
        }};
        const target = {data: [{}], clientWidth: 535, clientHeight: 480,
            _fullLayout: {width: 700, height: 480}};
        await app.fitUmapToContainer(target);
        await app.fitUmapToContainer(target);
        await app.fitUmapToContainer({...target, clientWidth: 0});
        console.log(JSON.stringify({calls, layout: target._fullLayout,
            autosize: app.umapLayout('test').autosize}));
    """)
    assert result == {"calls": 1, "layout": {"width": 535, "height": 480}, "autosize": True}


def test_compare_camera_preserves_rotation_and_existing_zoom_out(run_js):
    result = run_js("""
        app.spatial3dViewMode = 'compare';
        const original = {eye: {x: -1, y: 2, z: 1}, center: {x: 3, y: 0, z: 0},
            up: {x: 0, y: 1, z: 0}};
        const fitted = app.fittedSpatial3dCamera(original);
        const far = app.fittedSpatial3dCamera({eye: {x: 4, y: 4, z: 4}});
        console.log(JSON.stringify({original, fitted, far, reset: app.fittedSpatial3dCamera()}));
    """)
    eye = result["fitted"]["eye"]
    assert eye["y"] == pytest.approx(-2 * eye["x"])
    assert sum(value * value for value in eye.values()) == pytest.approx(3 * 1.8**2)
    assert result["fitted"]["center"] == {"x": 0, "y": 0, "z": 0}
    assert result["fitted"]["up"] == {"x": 0, "y": 1, "z": 0}
    assert result["original"]["center"]["x"] == 3
    assert result["far"]["eye"] == {"x": 4, "y": 4, "z": 4}
    assert result["reset"]["eye"]["x"] == pytest.approx(1.8)


def test_spatial3d_redraw_keeps_camera_and_does_not_mutate_plot_state(run_js):
    result = run_js("""
        const camera = {eye: {x: 1.8, y: -2, z: 1.2}, center: {x: 0.1, y: 0, z: 0},
            up: {x: 0, y: 0, z: 1}, projection: {type: 'perspective'}};
        const target = {_fullLayout: {scene: {camera}}};
        const layout = app.spatial3dLayout(target);
        const same = JSON.stringify(layout.scene.camera) === JSON.stringify(camera);
        layout.scene.camera.eye.x = 99;
        console.log(JSON.stringify({same, originalEyeX: camera.eye.x}));
    """)
    assert result == {"same": True, "originalEyeX": 1.8}


def test_numeric_labels_are_explicit_categories_after_significance_sort(run_js):
    result = run_js("""
        app.cauchyHeatmapData = [
            {annotation: 0, trait: '01', mlog10_p_cauchy: 5},
            {annotation: '10', trait: '01', mlog10_p_cauchy: 1},
            {annotation: 2, trait: '01', mlog10_p_cauchy: 8},
        ];
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify(plot));
    """)
    assert result["traces"][0]["y"] == ["10", "0", "2"]
    assert result["layout"]["yaxis"]["categoryarray"] == ["10", "0", "2"]
    assert result["layout"]["yaxis"]["type"] == result["layout"]["xaxis"]["type"] == "category"
    assert result["traces"][1]["zmin"] == -0.5
    assert result["traces"][1]["zmax"] == 2.5


def test_missing_results_stay_blank_and_do_not_bias_normalization(run_js):
    result = run_js("""
        app.cauchyHeatmapNormalize = true;
        app.cauchyHeatmapData = [
            {annotation: 'NA', trait: 't', mlog10_p_cauchy: null},
            {annotation: '01', trait: 't', mlog10_p_cauchy: 2},
            {annotation: '02', trait: 't', mlog10_p_cauchy: 4},
        ];
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify(plot.traces[0]));
    """)
    assert result["z"] == [[None], [0], [1]]
    assert result["customdata"] == [[None], [2], [4]]
    assert result["hoverongaps"] is False
    assert result["zmin"] == 0
    assert result["zmax"] == 1
    assert "0–1" in result["colorbar"]["title"]["text"]


def test_empty_selection_clears_old_heatmap(run_js):
    result = run_js("""
        app.cauchyHeatmapData = [];
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify(plot));
    """)
    assert result["purged"]
    assert result["cleared"]


def test_heatmap_paging_and_search_reuse_model_and_global_scale(run_js):
    result = run_js("""
        app.cauchyHeatmapNormalize = true;
        app.cauchyHeatmapAnnotationsPerPage = 1;
        app.cauchyHeatmapData = [
            {annotation: 'lowest', trait: 'IQ', mlog10_p_cauchy: 0},
            {annotation: 'middle', trait: 'IQ', mlog10_p_cauchy: 5},
            {annotation: 'highest', trait: 'IQ', mlog10_p_cauchy: 10},
        ];
        await app.renderCauchyHeatmap();
        const model = app.cauchyHeatmapModel();
        const first = plot.traces[0].z;
        app.cauchyHeatmapAnnotationPage = 1;
        await app.renderCauchyHeatmap();
        const second = {z: plot.traces[0].z, raw: plot.traces[0].customdata};
        app.cauchyHeatmapAnnotationQuery = 'lowest';
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify({first, second, last: plot.traces[0].z,
            max: plot.traces[0].zmax, reused: model === app.cauchyHeatmapModel(),
            annotationPage: app.cauchyHeatmapAnnotationPage}));
    """)
    assert result == dict(
        first=[[1]],
        second=dict(z=[[0.5]], raw=[[5]]),
        last=[[0]],
        max=1,
        reused=True,
        annotationPage=0,
    )


def test_large_sparse_heatmap_never_materializes_full_cartesian_matrix(run_js):
    result = run_js("""
        app.clusterData = () => {throw new Error('Unbounded exact clustering');};
        app.cauchyHeatmapData = Array.from({length: 10000}, (_, i) => ({
            annotation: 'annotation_' + i, trait: 'trait_' + i, mlog10_p_cauchy: i
        }));
        await app.renderCauchyHeatmap();
        const model = app.cauchyHeatmapModel();
        console.log(JSON.stringify({rows: plot.traces[0].z.length,
            cols: plot.traces[0].z[0].length, ordering: model.ordering,
            annotations: model.annotations.length, traits: model.traits.length,
            max: plot.traces[0].zmax, pages: app.cauchyHeatmapTraitPages}));
    """)
    assert result == dict(
        rows=40,
        cols=24,
        ordering="Fast pattern order (large matrix)",
        annotations=10000,
        traits=10000,
        max=9999,
        pages=417,
    )


def test_heatmap_all_pages_cover_every_label_and_transpose_preserves_values(run_js):
    result = run_js("""
        app.cauchyHeatmapCluster = false;
        app.cauchyHeatmapTraitsPerPage = 2;
        app.cauchyHeatmapAnnotationsPerPage = 2;
        app.cauchyHeatmapData = Array.from({length: 15}, (_, i) => ({
            annotation: 'a' + Math.floor(i / 5), trait: 't' + (i % 5),
            mlog10_p_cauchy: i
        }));
        await app.renderCauchyHeatmap();
        const seen = [];
        for (let a = 0; a < app.cauchyHeatmapAnnotationPages; a++) {
            for (let t = 0; t < app.cauchyHeatmapTraitPages; t++) {
                app.cauchyHeatmapAnnotationPage = a; app.cauchyHeatmapTraitPage = t;
                await app.renderCauchyHeatmap();
                seen.push(...plot.traces[0].z.flat());
            }
        }
        app.cauchyHeatmapTranspose = true;
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify({seen: seen.sort((a,b) => a-b),
            x: plot.traces[0].x, y: plot.traces[0].y, z: plot.traces[0].z,
            colorAxis: plot.traces[1].yaxis}));
    """)
    assert result == dict(seen=list(range(15)), x=["a0"], y=["t4"], z=[[4]], colorAxis="y2")


def test_heatmap_search_has_no_effect_on_enrichment_table_or_point_map_selection(run_js):
    result = run_js("""
        app.selectedTrait = 'IQ'; app.selectedAnnotationCategory = 'celltype';
        app.cauchyData = [{trait: 'IQ', type: 'aggregated', annotation_name: 'celltype',
            annotation: 'Brain', mlog10_p_median: 2}];
        app.cauchyHeatmapData = [{trait: 'IQ', annotation: 'Brain', mlog10_p_cauchy: 5}];
        app.cauchyHeatmapTraitQuery = 'missing';
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify({table: app.filteredCauchyTable.length,
            empty: app.cauchyHeatmapEmpty, cleared: plot.cleared, trait: app.selectedTrait}));
    """)
    assert result == dict(table=1, empty=True, cleared=True, trait="IQ")


def test_large_trait_and_sample_selectors_are_bounded_but_keyboard_reaches_all(run_js):
    result = run_js("""
        app.traits = Array.from({length: 2000}, (_, i) => 'Trait_' + i);
        app.samples = Array.from({length: 1000}, (_, i) => 'Slice_' + i);
        const before = [app.visibleTraitChoices.length, app.visibleSampleChoices.length];
        app.traitActiveIndex = 79; app.moveTraitSearch(1);
        app.sampleActiveIndex = 79; app.moveSampleSearch(1);
        const next = [app.visibleTraitChoices.length, app.visibleSampleChoices.length];
        app.traitSearchQuery = 'Trait_1999';
        console.log(JSON.stringify({before, next, found: app.filteredTraits[0]}));
    """)
    assert result == dict(before=[80, 80], next=[81, 81], found="Trait_1999")


def test_numeric_annotation_table_sort_search_and_legend_window(run_js):
    result = run_js("""
        app.selectedTrait = 'IQ'; app.selectedAnnotationCategory = 'domain';
        app.cauchyData = [0, 2, 10].map(annotation => ({trait: 'IQ', type: 'aggregated',
            annotation_name: 'domain', annotation, mlog10_p_median: annotation}));
        app.cauchyTableQuery = '10';
        app.getAnnotationSchema = () => ({categories: Array.from({length: 2000},
            (_, i) => 'domain' + i), colors: []});
        const initial = app.overviewDisplayedRows.length;
        app.overviewQuery = 'domain1999';
        console.log(JSON.stringify({table: app.filteredCauchyTable.map(r => r.annotation),
            initial, legend: app.overviewDisplayedRows.map(r => r.category)}));
    """)
    assert result == dict(table=[10], initial=80, legend=["domain1999"])


def test_heatmap_page_size_change_resets_plotly_zoom(run_js):
    result = run_js("""
        app.cauchyHeatmapData = Array.from({length: 80}, (_, i) => ({
            annotation: 'Brain', trait: 'Trait_' + i, mlog10_p_cauchy: i}));
        await app.renderCauchyHeatmap();
        const first = plot.layout.uirevision;
        app.cauchyHeatmapTraitsPerPage = 40;
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify({reset: first !== plot.layout.uirevision,
            traits: plot.traces[0].x.length}));
    """)
    assert result == dict(reset=True, traits=40)


def test_enrichment_annotation_sort_accepts_mixed_numeric_and_text_labels(run_js):
    result = run_js("""
        app.selectedTrait = 'IQ'; app.selectedAnnotationCategory = 'domain';
        app.cauchyData = [2, 10, '11', '01', 'Brain', null, null].map(annotation => ({
            trait: 'IQ', type: 'aggregated', annotation_name: 'domain', annotation}));
        app.cauchySortKey = 'annotation'; app.cauchySortAsc = true;
        const ascending = app.filteredCauchyTable.map(r => r.annotation);
        app.cauchySortAsc = false;
        console.log(JSON.stringify({ascending, descending: app.filteredCauchyTable.map(r => r.annotation)}));
    """)
    assert result == dict(
        ascending=["01", 2, 10, "11", "Brain", None, None],
        descending=["Brain", "11", 10, 2, "01", None, None],
    )


def test_heatmap_incremental_trait_and_annotation_subsets_are_independent(run_js):
    result = run_js("""
        app.scheduleCauchyHeatmap = () => {};
        app.cauchyHeatmapData = ['IQ','SCZ','Height'].flatMap((trait, i) =>
            ['Brain','Heart'].map((annotation,j) => ({trait,annotation,mlog10_p_cauchy:i+j})));
        app.cauchyHeatmapTraitQuery='IQ'; app.toggleHeatmapSubset('trait', app.heatmapSubsetChoices('trait')[0]);
        app.cauchyHeatmapTraitQuery='SCZ'; await app.renderCauchyHeatmap();
        const duringSearch=plot.traces[0].x.slice();
        app.toggleHeatmapSubset('trait','SCZ'); app.toggleHeatmapSubset('annotation','Brain');
        await app.renderCauchyHeatmap();
        const subset={traits:plot.traces[0].x.slice().sort(),annotations:plot.traces[0].y.slice()};
        app.cauchyHeatmapAnnotationQuery='Heart'; await app.renderCauchyHeatmap();
        const kept=plot.traces[0].y.slice();
        app.toggleHeatmapSubset('trait','IQ'); await app.renderCauchyHeatmap();
        const removed=plot.traces[0].x.slice();
        app.resetCauchyHeatmapFilters(); await app.renderCauchyHeatmap();
        console.log(JSON.stringify({duringSearch,subset,kept,removed,reset:[plot.traces[0].x.length,plot.traces[0].y.length]}));
    """)
    assert result == dict(
        duringSearch=["IQ"],
        subset=dict(traits=["IQ", "SCZ"], annotations=["Brain"]),
        kept=["Brain"],
        removed=["SCZ"],
        reset=[3, 2],
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_heatmap_hover_uses_annotation_color_and_original_value(run_js, transpose):
    result = run_js(
        """
        app.getAnnotationColor=(value,category) => value==='01' && category==='domain' ? '#00ff00' : '#000000';
        const transpose="""
        + json.dumps(transpose)
        + """;
        const details=app.heatmapHoverDetails({x:transpose?'01':'IQ <test>',y:transpose?'IQ <test>':'01',
            z:0.5,customdata:20,data:{meta:{heatmapRole:'matrix',transpose,metricLabel:'P',annotationName:'domain'}}});
        app.cauchyHeatmapTranspose=transpose;
        app.cauchyHeatmapData=[{trait:'IQ',annotation:'01',mlog10_p_cauchy:20}];
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify({details,stripAbove:plot.layout.yaxis2.domain[0]>plot.layout.yaxis.domain[1]}));
    """
    )
    assert result["details"] == dict(
        html="<b>01</b><br>IQ &lt;test&gt;<br>P: 20.00", background="#00ff00"
    )
    assert result["stripAbove"] is transpose


def test_heatmap_hover_survives_empty_search_and_does_not_leak_listeners(run_js):
    result = run_js("""
        const events=new Map(),dom=new Set(),globalEvents=new Set();
        plot.on=(name,fn)=>events.set(name,fn);
        plot.removeListener=(name,fn)=>{if(events.get(name)===fn)events.delete(name);};
        plot.addEventListener=(name,fn)=>dom.add(fn);
        plot.removeEventListener=(name,fn)=>dom.delete(fn);
        window.addEventListener=(name,fn)=>globalEvents.add(fn);
        window.removeEventListener=(name,fn)=>globalEvents.delete(fn);
        Plotly.purge=p=>{events.clear();delete p.data;};
        app.cauchyHeatmapData=[{trait:'IQ',annotation:'Brain',mlog10_p_cauchy:2}];
        await app.renderCauchyHeatmap();
        const initial={events:events.size,dom:dom.size,global:globalEvents.size};
        for(let i=0;i<3;i++) {
            app.cauchyHeatmapTraitQuery='no_match';await app.renderCauchyHeatmap();
            app.cauchyHeatmapTraitQuery='';await app.renderCauchyHeatmap();
        }
        console.log(JSON.stringify({initial,final:{events:events.size,dom:dom.size,global:globalEvents.size}}));
    """)
    assert result["initial"]["events"] == 3
    assert result["final"] == result["initial"]


def test_spatial2d_pan_adapter_preserves_data_and_updates_native_drag_ranges(run_js):
    result = run_js("""
        Plotly.version='3.6.0';
        const calls=[];
        const axis=range=>({type:'linear',range,autorange:false,setScale(){},_r:range.slice(),_rl:range.slice()});
        const scene={count:1,dirty:false,update:o=>calls.push(o),draw:()=>calls.push('draw'),selectBatch:[[]],unselectBatch:[[]]};
        const p={
            data:[{type:'scattergl',x:[1,2],y:[3,4],marker:{color:[0,1]}}],
            layout:{xaxis:{range:[0,10]},yaxis:{range:[10,0]}},
            _fullLayout:{
                xaxis:axis([0,10]),yaxis:axis([10,0]),_plots:{xy:{_scene:scene}},
                _glcanvas:{each:fn=>fn({regl:{clear:o=>calls.push(o)}})}
            }
        };
        const original=JSON.stringify(p.data);
        const moved=app.applySpatial2dPanViewport(p,{'xaxis.range':[5,15],'yaxis.range':[12,2]});
        const saved=p._fullLayout.xaxis._r.slice();
        const zoom=app.applySpatial2dPanViewport(p,{'xaxis.range':[5,10]});
        const reset=app.applySpatial2dPanViewport(p,{'xaxis.autorange':true});
        Plotly.version='4.0.0'; const unsupported=app.applySpatial2dPanViewport(p,{'xaxis.range':[6,16]});
        console.log(JSON.stringify({moved,saved,zoom,reset,unsupported,unchanged:original===JSON.stringify(p.data),calls}));
    """)
    assert result == dict(
        moved=True,
        saved=[5, 15],
        zoom=False,
        reset=False,
        unsupported=False,
        unchanged=True,
        calls=[dict(range=[5, 12, 15, 2]), dict(color=True, depth=True), "draw"],
    )


def test_spatial2d_pan_adapter_restores_axes_on_failure(run_js):
    result = run_js("""
        Plotly.version='3.6.0';
        let attempted=false;
        const axis=()=>({type:'linear',range:[0,10],_r:[0,10],_rl:[0,10],autorange:true,setScale(){}});
        const p={data:[{type:'scattergl'}],layout:{xaxis:{autorange:true},yaxis:{autorange:true}},
            _fullLayout:{xaxis:axis(),yaxis:axis(),_plots:{xy:{_scene:{count:1,draw(){},update(){attempted=true;throw Error('changed API')}}}},_glcanvas:{each(){}}}};
        const ok=app.applySpatial2dPanViewport(p,{'xaxis.range':[2,12]});
        console.log(JSON.stringify({ok,attempted,x:p._fullLayout.xaxis.range,r:p._fullLayout.xaxis._r,auto:p.layout.xaxis.autorange}));
    """)
    assert result == dict(ok=False, attempted=True, x=[0, 10], r=[0, 10], auto=True)


def test_spatial2d_sync_keeps_latest_request_during_slow_relayout(run_js):
    result = run_js("""
        const waiting=[],calls=[]; const target={data:[{}]};
        Plotly.relayout=(p,v)=>{calls.push(v); return new Promise(resolve=>waiting.push(resolve));};
        const first=app.queueSpatial2dViewport(target,{'xaxis.range':[1,11]});
        app.queueSpatial2dViewport(target,{'xaxis.autorange':true});
        app.queueSpatial2dViewport(target,{'xaxis.range':[3,13]});
        waiting.shift()(); await Promise.resolve();
        waiting.shift()(); await first;
        console.log(JSON.stringify({calls,busy:target.__gsMapApplyingViewport,pending:target.__gsMapPendingViewport}));
    """)
    assert result == dict(
        calls=[{"xaxis.range": [1, 11]}, {"xaxis.range": [3, 13]}], busy=False, pending=None
    )


def test_spatial2d_sync_merges_partial_ranges_without_conflicting_plotly_keys(run_js):
    result = run_js("""
        const waiting=[],calls=[]; const target={data:[{}]};
        Plotly.relayout=(p,v)=>{calls.push(v);return new Promise(resolve=>waiting.push(resolve));};
        const first=app.queueSpatial2dViewport(target,{'xaxis.range':[0,10]});
        app.queueSpatial2dViewport(target,{'xaxis.range':[1,11]});
        app.queueSpatial2dViewport(target,{'xaxis.range[0]':2});
        app.queueSpatial2dViewport(target,{'xaxis.autorange':false});
        waiting.shift()();await Promise.resolve();waiting.shift()();await first;
        console.log(JSON.stringify(calls));
    """)
    assert result == [{"xaxis.range": [0, 10]}, {"xaxis.range": [2, 11], "xaxis.autorange": False}]


def test_spatial3d_sync_reads_current_camera_and_preserves_last_frame(run_js):
    result = run_js("""
        const frames=new Map(),calls=[];let frameId=0;
        global.requestAnimationFrame=fn=>{frames.set(++frameId,fn);return frameId;};
        global.cancelAnimationFrame=id=>frames.delete(id);
        const tick=()=>{const batch=[...frames.values()];frames.clear();batch.forEach(fn=>fn());};
        const panel=id=>({id,data:[{}],handlers:{},on(name,fn){this.handlers[name]=fn;}});
        const a=panel('a'),b=panel('b');document.getElementById=id=>({a,b}[id]);
        const currentScene={getCamera:()=>({eye:{x:7}})};
        a._fullLayout={scene:{_scene:currentScene}};
        app.queueSpatial3dCamera=(p,v)=>calls.push({target:p.id,camera:v});
        app.bindSpatial3dCameraSync('a','b');app.bindSpatial3dCameraSync('b','a');
        a.handlers.plotly_relayouting({'scene.camera':{eye:{x:1}}});
        a.handlers.plotly_relayouting({'scene.camera':{eye:{x:2}}});
        const frameCount=frames.size;tick();
        b.handlers.plotly_relayouting({'scene.camera':{eye:{x:3}}});tick();
        a.handlers.plotly_relayouting({'scene.camera':{eye:{x:4}}});
        a.handlers.plotly_relayout({'scene.camera':{eye:{x:5}}});tick();
        a.__gsMapApplyingCamera=true;a.handlers.plotly_relayout({'scene.camera':{eye:{x:100}}});
        a.__gsMapApplyingCamera=false;app._spatial3dSyncing=true;
        a.handlers.plotly_relayouting({'scene.camera':{eye:{x:101}}});tick();
        app._spatial3dSyncing=false;app.spatial3dViewMode='trait';
        a.handlers.plotly_relayout({'scene.camera':{eye:{x:102}}});
        console.log(JSON.stringify({frameCount,calls}));
    """)
    assert result == dict(
        frameCount=1,
        calls=[
            {"target": "b", "camera": {"eye": {"x": 7}}},
            {"target": "a", "camera": {"eye": {"x": 3}}},
            {"target": "b", "camera": {"eye": {"x": 5}}},
        ],
    )


def test_spatial3d_camera_queue_coalesces_slow_updates_and_recovers_on_error(run_js):
    result = run_js("""
        const waiting=[],calls=[],target={data:[{}]};
        Plotly.relayout=(p,v)=>{calls.push(v);return new Promise(resolve=>waiting.push(resolve));};
        const first=app.queueSpatial3dCamera(target,{eye:{x:1}});
        app.queueSpatial3dCamera(target,{eye:{x:2}});
        const camera={eye:{x:3}};app.queueSpatial3dCamera(target,camera);camera.eye.x=99;
        waiting.shift()();await Promise.resolve();waiting.shift()();await first;
        Plotly.relayout=()=>Promise.reject(new Error('context lost'));
        await app.queueSpatial3dCamera(target,{eye:{x:4}});
        const failedClean=!target.__gsMapApplyingCamera && !target.__gsMapPendingCamera;
        Plotly.relayout=async(p,v)=>calls.push(v);
        await app.queueSpatial3dCamera(target,{eye:{x:5}});
        console.log(JSON.stringify({calls,failedClean,busy:target.__gsMapApplyingCamera}));
    """)
    assert result == dict(
        calls=[{"scene.camera": {"eye": {"x": value}}} for value in [1, 3, 5]],
        failedClean=True,
        busy=False,
    )


def test_spatial3d_empty_wheel_event_tracks_camera_tail_then_stops(run_js):
    result = run_js("""
        const frames=new Map(),events={},dom={},calls=[];let frameId=0,x=9;
        global.requestAnimationFrame=fn=>{frames.set(++frameId,fn);return frameId;};
        global.cancelAnimationFrame=id=>frames.delete(id);
        const tick=()=>{const batch=[...frames.values()];frames.clear();batch.forEach(fn=>fn());};
        const scene={getCamera:()=>({eye:{x,y:0,z:1},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1}})};
        const source={on:(k,f)=>events[k]=f,addEventListener:(k,f)=>dom[k]=f,
            _fullLayout:{scene:{_scene:scene}}};
        const target={data:[{}]};document.getElementById=id=>id==='source'?source:target;
        app.queueSpatial3dCamera=(p,c)=>calls.push(c.eye.x);
        app.bindSpatial3dCameraSync('source','target');
        events.plotly_relayout({});dom.wheel();tick();x=10;tick();tick();tick();
        console.log(JSON.stringify({calls,pendingFrames:frames.size}));
    """)
    assert result == dict(calls=[9, 10], pendingFrames=0)


def test_spatial2d_pan_sync_is_live_bidirectional_and_coalesces_frames(run_js):
    result = run_js("""
        const frames=new Map(),calls=[],queued=[];let frameId=0;
        global.requestAnimationFrame=fn=>{frames.set(++frameId,fn);return frameId;};
        global.cancelAnimationFrame=id=>frames.delete(id);
        const tick=()=>{const batch=[...frames.values()];frames.clear();batch.forEach(fn=>fn());};
        const panel=id=>({id,data:[{}],handlers:{},on(name,fn){this.handlers[name]=fn;}});
        const a=panel('a'),b=panel('b');document.getElementById=id=>({a,b}[id]);
        app.applySpatial2dPanViewport=(p,v)=>{calls.push({target:p.id,viewport:v});return !v['xaxis.autorange'];};
        app.queueSpatial2dViewport=(p,v)=>queued.push({target:p.id,viewport:v});
        app.bindSpatial2dViewportSync('a','b');app.bindSpatial2dViewportSync('b','a');
        a.handlers.plotly_relayouting({'xaxis.range':[1,11]});
        a.handlers.plotly_relayouting({'xaxis.range':[2,12]});
        const queuedFrames=frames.size;tick();const beforeRelease=calls.length;
        b.handlers.plotly_relayouting({'xaxis.range':[3,13]});tick();
        a.handlers.plotly_relayouting({'xaxis.range':[4,14]});
        a.handlers.plotly_relayout({'xaxis.range':[5,15]});tick();
        a.handlers.plotly_relayouting({'xaxis.autorange':true});tick();const previewQueued=queued.length;
        a.handlers.plotly_relayout({'xaxis.autorange':true});
        a.__gsMapApplyingViewport=true;a.handlers.plotly_relayout({'xaxis.range':[99,109]});
        a.__gsMapApplyingViewport=false;app.spatial2dViewMode='annotation';
        a.handlers.plotly_relayouting({'xaxis.range':[100,110]});tick();
        console.log(JSON.stringify({queuedFrames,beforeRelease,previewQueued,calls,queued}));
    """)
    assert result["queuedFrames"] == result["beforeRelease"] == 1
    assert result["previewQueued"] == 0
    assert result["calls"] == [
        {"target": "b", "viewport": {"xaxis.range": [2, 12]}},
        {"target": "a", "viewport": {"xaxis.range": [3, 13]}},
        {"target": "b", "viewport": {"xaxis.range": [5, 15]}},
        {"target": "b", "viewport": {"xaxis.autorange": True}},
        {"target": "b", "viewport": {"xaxis.autorange": True}},
    ]
    assert result["queued"] == [{"target": "b", "viewport": {"xaxis.autorange": True}}]


def test_discrete_scale_matches_half_integer_color_limits(run_js):
    scales = run_js("""
        console.log(JSON.stringify(Array.from({length: 64}, (_, i) =>
            app.discreteColorScale(Array.from({length: i + 1}, (_, j) => 'color' + j))
        )));
    """)
    for count, scale in enumerate(scales, start=1):
        for index in range(count):
            position = (index + 0.5) / count
            lower = next(color for bound, color in reversed(scale) if bound <= position)
            upper = next(color for bound, color in scale if bound >= position)
            assert lower == upper == f"color{index}"


def test_enrichment_load_failure_clears_stale_results_and_explains_error(run_js):
    result = run_js("""
        app.cauchyIndex = {table: {}, heatmap: {}};
        app.cauchyData = [{annotation: 'old'}];
        app.cauchyHeatmapData = [{annotation: 'old'}];
        app.loadCauchyShard = () => Promise.reject(new Error('Missing data file'));
        try {await app.loadActiveCauchyData();} catch (_) {}
        console.log(JSON.stringify({
            error: app.cauchyError, loading: app.loadingCauchyData,
            rows: app.cauchyData, heatmap: app.cauchyHeatmapData, purged: plot.purged
        }));
    """)
    assert result == {
        "error": "Missing data file",
        "loading": False,
        "rows": [],
        "heatmap": [],
        "purged": True,
    }


def test_clustering_with_sparse_trait_annotation_pairs_keeps_gaps(run_js):
    result = run_js("""
        app.cauchyHeatmapData = [
            {annotation: 'a', trait: 't1', mlog10_p_cauchy: 2},
            {annotation: 'b', trait: 't1', mlog10_p_cauchy: 4},
            {annotation: 'a', trait: 't2', mlog10_p_cauchy: 3},
        ];
        await app.renderCauchyHeatmap();
        console.log(JSON.stringify(plot.traces[0]));
    """)
    assert sorted(value for row in result["z"] for value in row if value is not None) == [2, 3, 4]
    assert sum(value is None for row in result["z"] for value in row) == 1


def test_trait_switch_reuses_heatmap_but_changed_view_data_or_missing_plot_redraw(run_js):
    result = run_js("""
        let rows = [{annotation: '01', trait: 'IQ', mlog10_p_cauchy: 2}];
        app.selectedAnnotationCategory = 'domain';
        app.cauchyIndex = {table: {all: {IQ: {domain: {table: true}},
            AF: {domain: {table: true}}}}};
        app.loadCauchyShard = async entry => entry?.table ? [{trait: app.selectedTrait}] : rows;
        app.selectedTrait = 'IQ';
        await app.loadActiveCauchyData();
        app.selectedTrait = 'AF';
        await app.loadActiveCauchyData();
        const same = {renders: plot.renderCount, table: app.cauchyData, matches: app.cauchyHeatmapSelectionMatches};
        app.cauchyViewMode = 'sample';
        const beforeViewLoad = app.cauchyHeatmapSelectionMatches;
        await app.loadActiveCauchyData();
        rows = [...rows, {annotation: '02', trait: 'IQ', mlog10_p_cauchy: 3}];
        await app.loadActiveCauchyData();
        Plotly.purge(plot);
        await app.loadActiveCauchyData();
        console.log(JSON.stringify({same, beforeViewLoad, renders: plot.renderCount}));
    """)
    assert result == {
        "same": {"renders": 1, "table": [{"trait": "AF"}], "matches": True},
        "beforeViewLoad": False,
        "renders": 4,
    }


def test_same_trait_requested_again_cannot_be_overwritten_by_older_request(run_js):
    result = run_js("""
        const pending = [];
        app.cauchyIndex = {table: {all: {A: {domain: {table: true}}, B: {domain: {table: true}}}}};
        app.selectedAnnotationCategory = 'domain';
        app.loadCauchyShard = entry => entry?.table
            ? new Promise(resolve => pending.push(resolve))
            : Promise.resolve([{annotation: '01', trait: 'A', mlog10_p_cauchy: 2}]);
        app.selectedTrait = 'A';
        const first = app.loadActiveCauchyData();
        app.selectedTrait = 'B';
        const second = app.loadActiveCauchyData();
        app.selectedTrait = 'A';
        const last = app.loadActiveCauchyData();
        pending[2]([{trait: 'A', version: 'latest'}]);
        await last;
        pending[0]([{trait: 'A', version: 'stale'}]);
        pending[1]([{trait: 'B', version: 'stale'}]);
        await Promise.all([first, second]);
        console.log(JSON.stringify({rows: app.cauchyData, renders: plot.renderCount, loading: app.loadingCauchyData}));
    """)
    assert result == {
        "rows": [{"trait": "A", "version": "latest"}],
        "renders": 1,
        "loading": False,
    }


@pytest.mark.parametrize("mode", ["trait", "annotation", "compare"])
def test_trait_only_render_leaves_annotation_panels_unchanged(run_js, mode):
    result = run_js(
        """
        app.spatial2dViewMode = app.spatial3dViewMode = """
        + json.dumps(mode)
        + """;
        app.plotlyReady = true;
        app.currentSampleData = {};
        app.spatial3dBase = {};
        const calls = [];
        app.renderSpatial2DPlotSingle = (div, type) => calls.push('2d-' + type);
        app.renderSpatial3dTrait = () => calls.push('3d-trait');
        app.renderSpatial3dAnnotation = () => calls.push('3d-annotation');
        app.renderSpatial2DPlot({traitOnly: true});
        await app.renderSpatial3dViews({traitOnly: true});
        const traitOnly = [...calls];
        calls.length = 0;
        app.renderSpatial2DPlot();
        await app.renderSpatial3dViews();
        console.log(JSON.stringify({traitOnly, all: calls}));
    """
    )
    assert result["traitOnly"] == ([] if mode == "annotation" else ["2d-trait", "3d-trait"])
    expected = [
        f"{dimension}-{kind}"
        for dimension in ("2d", "3d")
        for kind in ("trait", "annotation")
        if mode == "compare" or kind == mode
    ]
    assert result["all"] == expected


def test_overview_sort_metrics_default_to_cauchy_and_keep_missing_last(run_js):
    result = run_js("""
        app.overviewData={rows:[
            {category:'10',cauchy_logp:null,median_logp:2,q95_logp:3,n_spots:10},
            {category:'2',cauchy_logp:0,median_logp:4,q95_logp:2,n_spots:30},
            {category:'1',cauchy_logp:5,median_logp:null,q95_logp:1,n_spots:20}
        ]};
        const result={defaultSort:app.overviewSort};
        for (const key of ['cauchy_logp','median_logp','q95_logp','n_spots','category']) {
            app.overviewSort=key;
            result[key]=app.overviewVisibleRows.map(row=>row.category);
        }
        result.original=app.overviewData.rows.map(row=>row.category);
        console.log(JSON.stringify(result));
    """)
    assert result == {
        "defaultSort": "cauchy_logp",
        "cauchy_logp": ["1", "2", "10"],
        "median_logp": ["2", "10", "1"],
        "q95_logp": ["10", "2", "1"],
        "n_spots": ["2", "1", "10"],
        "category": ["1", "2", "10"],
        "original": ["10", "2", "1"],
    }


def test_trait_change_refreshes_threshold_filtered_annotation_comparison(run_js):
    result = run_js("""
        app.plotlyReady=true;app.currentSampleData={};app.spatial3dBase={};
        app.spatial2dAssociationMinLogP=app.spatial3dAssociationMinLogP=5;
        const calls=[];
        app.renderSpatial2DPlotSingle=(div,type)=>calls.push('2d-'+type);
        app.renderSpatial3dTrait=()=>calls.push('3d-trait');
        app.renderSpatial3dAnnotation=()=>calls.push('3d-annotation');
        app.renderSpatial2DPlot({traitOnly:true});await app.renderSpatial3dViews({traitOnly:true});
        console.log(JSON.stringify(calls));
    """)
    assert result == ["2d-trait", "2d-annotation", "3d-trait", "3d-annotation"]


def test_spatial3d_opacity_changes_skip_plotly_scene_rebuilds(run_js):
    result = run_js("""
        const n=500, view={id:'spatial3dPlotTrait',on(){},addEventListener(){}};
        document.getElementById=id=>id===view.id?view:null;
        app.plotlyReady=true;app.bindSpatial3dHover=()=>{};app.bindSpatial3dCameraSync=()=>{};
        const x=Float32Array.from({length:n},(_,i)=>i%17),y=Float32Array.from({length:n},(_,i)=>i%5),
            z=Float32Array.from({length:n},(_,i)=>i%3);
        app.spatial3dBase={x,y,z,positions:new Float32Array(n*3)};
        app.spatial3dTraitData={values:Float32Array.from({length:n},(_,i)=>i%11),cmin:0,cmax:10,q95:9};
        app.spatial3dViewMode='trait';
        await app.renderSpatial3dViews();
        view._fullLayout={scene:{_scene:{},camera:{eye:{x:1,y:1,z:1}}}};
        const counts=[view.renderCount];
        app.spatial3dOpacity=1;await app.renderSpatial3dViews();
        counts.push(view.renderCount);
        const opaque=view.__gsMap3dPoints.opaque;
        app.spatial3dPointSize=Number(app.spatial3dPointSize)+1;await app.renderSpatial3dViews();
        counts.push(view.renderCount);
        app.spatial3dTraitData={...app.spatial3dTraitData,cmax:20};await app.renderSpatial3dViews();
        counts.push(view.renderCount);
        delete view._fullLayout;app.spatial3dOpacity=.5;await app.renderSpatial3dViews();
        counts.push(view.renderCount);
        console.log(JSON.stringify({counts,opaque}));
    """)
    # Opacity alone: no rebuild. Point size, colour range and a purged scene: rebuild.
    assert result == {"counts": [1, 1, 2, 3, 4], "opaque": True}
