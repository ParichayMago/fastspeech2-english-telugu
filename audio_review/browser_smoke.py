"""Optional Playwright smoke test; synthetic judgments go only to a temp store."""
import csv
import io
import json
from pathlib import Path
import tempfile
import threading
from http.server import ThreadingHTTPServer

from server import ROOT,ReviewStore,PairReviewStore,build_catalog,handler_for


def main():
    from playwright.sync_api import sync_playwright
    screenshots=ROOT/'tmp/listening-review-ui';screenshots.mkdir(parents=True,exist_ok=True)
    catalog=build_catalog(ROOT);errors=[]
    with tempfile.TemporaryDirectory(prefix='synthetic-browser-reviews-',dir=ROOT/'tmp') as temporary:
        reviews=ReviewStore(Path(temporary)/'individual.json');pairs=PairReviewStore(Path(temporary)/'pairs.json')
        server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(catalog,reviews,'synthetic-ui-test',ROOT,pairs))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with sync_playwright() as runtime:
                browser=runtime.chromium.launch(channel='chromium',headless=True,args=['--autoplay-policy=no-user-gesture-required','--disable-gpu'])
                page=browser.new_page(viewport=dict(width=1440,height=1100),accept_downloads=True)
                page.on('pageerror',lambda error:errors.append(str(error)))
                url='http://127.0.0.1:'+str(server.server_port)
                page.goto(url+'/?view=final');page.wait_for_function("document.querySelector('#matchCount').textContent==='10 matches'")
                assert page.locator('#list [data-id]').count()==10
                page.select_option('#language','telugu');assert page.locator('#list [data-id]').count()==5
                page.select_option('#language','english');assert page.locator('#list [data-id]').count()==5
                page.fill('#notes','SYNTHETIC UI CHECK; not a listening judgment.')
                page.click('[data-status="needs_work"]');page.wait_for_function("state.dirty.size===0 && state.loading===0")
                page.reload();page.wait_for_function("document.querySelector('#matchCount').textContent==='10 matches'")
                assert page.input_value('#notes')=='SYNTHETIC UI CHECK; not a listening judgment.'
                with page.expect_download() as download:
                    page.click('#allExport')
                rows=list(csv.DictReader(io.StringIO(Path(download.value.path()).read_text(encoding='utf-8-sig'))))
                assert len(rows)==1 and rows[0]['status']=='needs_work'
                page.click('[data-kind="comparison"]');assert page.locator('#pairSelect option').count()==6
                assert page.locator('#pairForm audio').count()==2
                assert page.input_value('#preference-overall')==''
                page.locator('body').press('Space')
                assert page.locator('audio').evaluate_all('(audios)=>audios.every(a=>a.paused)'), 'Hidden player started from comparison keyboard shortcut'
                first,second=page.locator('#pairForm audio').all()
                first.evaluate('(audio)=>audio.play()');page.wait_for_function("!document.querySelector('#pairForm audio').paused")
                second.evaluate('(audio)=>audio.play()');page.wait_for_function("document.querySelectorAll('#pairForm audio')[0].paused && !document.querySelectorAll('#pairForm audio')[1].paused")
                page.locator('#pairForm audio').evaluate_all('(audios)=>audios.forEach(a=>a.pause())')
                for variant in ('A','B'):page.fill('#notes-'+variant,'SYNTHETIC '+variant+' note; not a listening judgment.')
                for key,value in [('clarity','A'),('pacing','B'),('intonation','same'),('overall','unsure')]:page.select_option('#preference-'+key,value)
                page.fill('#pairReviewer','Synthetic browser fixture');page.check('#listenedBoth');page.click('#savePair')
                page.wait_for_function("document.querySelector('#pairStatus').textContent==='Saved on this Mac'")
                page.goto(url+'/?view=comparison');page.wait_for_function("document.querySelector('#pairStatus').textContent==='Saved on this Mac'")
                assert page.input_value('#preference-overall')=='unsure'
                assert page.input_value('#notes-A').startswith('SYNTHETIC A')
                with page.expect_download() as download:page.click('#exportPairs')
                exported=json.loads(Path(download.value.path()).read_text())
                assert len(exported)==1 and exported[0]['preferences']['overall']=='unsure'
                page.select_option('#pairSelect','E2a');assert page.input_value('#preference-overall')==''
                page.fill('#notes-A','Unsaved synthetic draft')
                page.select_option('#pairSelect','E1');page.select_option('#pairSelect','E2a')
                assert page.input_value('#notes-A')=='Unsaved synthetic draft'
                # Use another empty temporary store for clean screenshots, never the user's ratings.
                clean_server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(catalog,
                    ReviewStore(Path(temporary)/'empty.json'),'screenshot-only',ROOT,
                    PairReviewStore(Path(temporary)/'empty-pairs.json')))
                clean_thread=threading.Thread(target=clean_server.serve_forever,daemon=True);clean_thread.start()
                clean_url='http://127.0.0.1:'+str(clean_server.server_port)
                clean=browser.new_page(viewport=dict(width=1440,height=1100))
                clean.on('pageerror',lambda error:errors.append(str(error)))
                try:
                    clean.goto(clean_url+'/?view=comparison')
                    clean.wait_for_function("document.querySelector('#pairForm audio')!==null")
                    assert clean.locator('#pairProgress').inner_text()=='0 of 6 sentence comparisons saved'
                    clean.screenshot(path=str(screenshots/'comparison-desktop.png'),full_page=True)
                    clean.set_viewport_size(dict(width=390,height=844))
                    assert clean.evaluate('document.documentElement.scrollWidth<=innerWidth'), 'Mobile horizontal overflow'
                    clean.screenshot(path=str(screenshots/'comparison-mobile.png'))
                    clean.locator('#savePair').scroll_into_view_if_needed()
                    clean.screenshot(path=str(screenshots/'comparison-mobile-form.png'))
                    clean.goto(clean_url+'/?view=final')
                    clean.wait_for_function("document.querySelector('#matchCount').textContent==='10 matches'")
                    clean.set_viewport_size(dict(width=1440,height=1100))
                    clean.screenshot(path=str(screenshots/'final-recordings.png'),full_page=True)
                finally:
                    clean_server.shutdown();clean_server.server_close();clean_thread.join()
                assert not errors,errors
                browser.close()
        finally:
            server.shutdown();server.server_close();thread.join()
    print(json.dumps(dict(result='passed',final_recordings=10,english_pairs=6,
        playback_and_single_active_player=True,save_reload_export=True,draft_retention=True,
        mobile_no_horizontal_overflow=True,javascript_errors=errors,
        synthetic_ratings_isolated_and_deleted=True,actual_listener_judgments_created=False),indent=2))


if __name__=='__main__':main()
