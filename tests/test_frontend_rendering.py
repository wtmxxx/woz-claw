from pathlib import Path


def test_frontend_does_not_render_memory_hits_hint() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "memory_hits:" not in html


def test_frontend_layout_is_not_centered_container() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "justify-content: center;" not in html
    assert "align-items: center;" not in html


def test_frontend_messages_have_horizontal_margin() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "margin-inline:" in html


def test_frontend_messages_use_readable_sans_font() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "font-family: \"PingFang SC\", \"Hiragino Sans GB\", \"Source Han Sans SC\", \"Microsoft YaHei UI\", sans-serif;" in html


def test_frontend_has_global_ui_scale() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "--ui-scale: 1.15;" in html
    assert "transform: scale(var(--ui-scale));" in html
    assert "width: calc(100vw / var(--ui-scale));" in html
    assert "height: calc(100vh / var(--ui-scale));" in html


def test_frontend_locks_root_scroll_container() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "html," in html
    assert "width: 100%;" in html
    assert "height: 100%;" in html
    assert "overflow: hidden;" in html


def test_frontend_resets_scroll_on_refresh() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "history.scrollRestoration = 'manual';" in html
    assert "window.scrollTo(0, 0);" in html
    assert "requestAnimationFrame(() => {" in html
    assert "window.addEventListener('pageshow', scheduleScrollReset);" in html
    assert "window.addEventListener('load', scheduleScrollReset);" in html
    assert "window.addEventListener('beforeunload', resetPageScroll);" in html


def test_frontend_app_is_pinned_to_top_left_viewport() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert ".app {" in html
    assert "position: fixed;" in html
    assert "top: 0;" in html
    assert "left: 0;" in html


def test_frontend_composer_uses_fixed_bar_height() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    composer_start = html.index(".composer {")
    composer_end = html.index("}", composer_start)
    composer_block = html[composer_start:composer_end]

    composer_input_start = html.index(".composer input {")
    composer_input_end = html.index("}", composer_input_start)
    composer_input_block = html[composer_input_start:composer_input_end]

    assert "height: var(--composer-bar-height);" in composer_block
    assert "height: 44px;" in composer_input_block


def test_frontend_sidebar_has_no_bottom_padding_for_footer_alignment() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    sidebar_start = html.index(".sidebar {")
    sidebar_end = html.index("}", sidebar_start)
    sidebar_block = html[sidebar_start:sidebar_end]

    assert "padding: 18px 18px 0;" in sidebar_block


def test_frontend_has_tool_call_rendering() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "tool_calls" in html
    assert "调用轨迹" in html
    assert "loaded_skills" in html
    assert "activity_traces" in html


def test_frontend_tool_calls_are_collapsible_and_above_ai_message() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "createElement('details')" in html
    assert "summary.textContent = `调用轨迹" in html
    assert "addAssistantMsg" in html


def test_frontend_history_load_uses_persisted_tool_calls() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "item.tool_calls || []" in html
    assert "item.loaded_skills || []" in html
    assert "item.activity_traces || []" in html


def test_frontend_tool_calls_have_visible_separator_between_items() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "lines.push('--------------------');" in html


def test_frontend_tool_trace_left_aligns_with_assistant_message() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert ".assistant-block .msg {" in html
    assert "margin-inline: 0;" in html


def test_frontend_sidebar_has_reserved_bottom_space_with_scrollbar() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    root_start = html.index(":root {")
    root_end = html.index("}", root_start)
    root_block = html[root_start:root_end]

    sidebar_start = html.index(".sidebar {")
    sidebar_end = html.index("}", sidebar_start)
    sidebar_block = html[sidebar_start:sidebar_end]

    conv_start = html.index(".conv-list {")
    conv_end = html.index("}", conv_start)
    conv_list_block = html[conv_start:conv_end]

    assert "--composer-bar-height: 64px;" in root_block
    assert "grid-template-rows: auto auto minmax(0, 1fr) var(--composer-bar-height);" in sidebar_block
    assert "overflow: auto;" in conv_list_block
    assert ".sidebar-footer {" in html
    assert "height: var(--composer-bar-height);" in html


def test_frontend_includes_favicon_icon() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert 'rel="icon"' in html
    assert "https://www.wotemo.com/img/WotemoRoundCorner.png" in html


def test_frontend_has_bottom_left_settings_and_user_profile() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert 'id="openSettingsBtn"' in html
    assert 'id="userProfile"' in html
    assert 'id="userAvatar"' in html
    assert 'id="userNameLabel"' in html


def test_frontend_new_conversation_button_uses_dedicated_style_class() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    btn_start = html.index(".new-conv-btn {")
    btn_end = html.index("}", btn_start)
    btn_block = html[btn_start:btn_end]

    assert 'id="newConversationBtn" class="new-btn new-conv-btn' in html
    assert ".new-conv-btn {" in html
    assert "box-shadow: none;" in btn_block
    assert "background: #fff;" in btn_block
    assert "color: #1f56d8;" in btn_block


def test_frontend_new_conversation_button_is_vertically_centered() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    btn_start = html.index(".new-conv-btn {")
    btn_end = html.index("}", btn_start)
    btn_block = html[btn_start:btn_end]

    assert "align-items: flex-start;" not in btn_block
    assert "align-items:center;" in btn_block


def test_frontend_new_conversation_button_content_is_horizontally_centered() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    btn_start = html.index(".new-conv-btn {")
    btn_end = html.index("}", btn_start)
    btn_block = html[btn_start:btn_end]

    assert "justify-content: flex-start;" not in btn_block
    assert "justify-content:center;" in btn_block


def test_frontend_settings_button_is_semi_iconized() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert 'id="openSettingsBtn"' in html
    assert "settings-btn-icon" in html
    assert "settings-btn-label" in html
    assert "⚙" in html
    assert "gap: 3px;" in html
    assert ".settings-btn-icon {" in html
    assert "font-size: 13px;" in html


def test_frontend_has_settings_view_for_memory_and_skills() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert 'id="settingsView"' in html
    assert 'id="longTermMemory"' in html
    assert 'id="skillsList"' in html
    assert 'id="saveSettingsBtn"' in html
    assert 'id="backToChatBtn"' in html


def test_frontend_has_skill_upload_controls() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert 'id="skillUploadFile"' in html
    assert 'id="skillUploadBtn"' in html
    assert "上传 Skill ZIP" in html


def test_frontend_memory_textarea_uses_full_panel_height_layout() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "grid-template-rows: 1fr;" in html
    assert "#settingsPanelMemory {" in html
    assert "#settingsPanelMemory .settings-group {" in html
    assert "#longTermMemory {" in html
    assert "height: 100%;" in html


def test_frontend_settings_panel_wrap_keeps_scroll_for_skills() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    panel_wrap_start = html.index(".settings-panel-wrap {")
    panel_wrap_end = html.index("}", panel_wrap_start)
    panel_wrap_block = html[panel_wrap_start:panel_wrap_end]

    assert "overflow: auto;" in panel_wrap_block


def test_frontend_avatar_uses_initials_of_each_word() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "function getUserInitials" in html
    assert "replace(/([a-z])([A-Z])/g, '$1 $2')" in html
    assert "split(/[^a-zA-Z0-9]+/)" in html
    assert "slice(0, 2)" in html


def test_frontend_choice_custom_submit_button_prevents_text_wrap() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "customInput.className = 'flex-1 min-w-0 border border-slate-200 rounded-xl bg-white px-3 py-2 text-sm';" in html
    assert "customSubmit.className = 'secondary-btn shrink-0 whitespace-nowrap min-w-24';" in html


def test_frontend_choice_submit_renders_user_choice_message() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "const finalChoice = customInputText || selectedOption;" in html
    assert "const choiceMessage = question" in html
    assert "const pendingChoiceMsgEl = addMsg('u', choiceMessage, false);" in html
    assert "pendingChoiceMsgEl.dataset.pendingChoice = '1';" in html
    assert "pendingChoiceMsgEl.remove();" in html


def test_frontend_approval_submit_renders_status_and_tracks_completed_requests() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "let completedRequestIds = new Set();" in html
    assert "function markRequestCompleted(requestId)" in html
    assert "function unmarkRequestCompleted(requestId)" in html
    assert "function removeRequestCardsById(type, requestId)" in html
    assert "function clearTransientStatusMessages()" in html
    assert "function addTransientStatusMsg(text)" in html
    assert "if (completedRequestIds.has(requestId)) return;" in html
    assert "const approvedLabel = approved ? '已批准' : '已拒绝';" in html
    assert "addTransientStatusMsg(approvedLabel);" in html
    assert "wrap.remove();" in html
    assert "loadCompletedRequestIds();" in html
    assert "removeRequestCardsById('approval', requestId);" in html
    assert "removeRequestCardsById('choice', requestId);" in html


def test_frontend_conversation_list_supports_delete_action() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    trigger_start = html.index(".conv-menu-trigger {")
    trigger_end = html.index("}", trigger_start)
    trigger_block = html[trigger_start:trigger_end]
    trigger_hover_start = html.index(".conv-menu-trigger:hover {")
    trigger_hover_end = html.index("}", trigger_hover_start)
    trigger_hover_block = html[trigger_hover_start:trigger_hover_end]
    item_hover_start = html.index(".conv-item:hover {")
    item_hover_end = html.index("}", item_hover_start)
    item_hover_block = html[item_hover_start:item_hover_end]
    item_active_start = html.index(".conv-item.active {")
    item_active_end = html.index("}", item_active_start)
    item_active_block = html[item_active_start:item_active_end]

    assert "className = 'conv-card'" in html
    assert ".conv-card {" in html
    assert "border-radius: 10px;" in html
    assert "className = 'conv-menu-trigger'" in html
    assert "textContent = '⋯'" in html
    assert "background: transparent;" in trigger_block
    assert "border: none;" in trigger_block
    assert "background: #ffffff;" not in trigger_block
    assert "background: transparent;" in trigger_hover_block
    assert "background: #eef3fb;" not in trigger_hover_block
    assert "background: transparent;" in item_hover_block
    assert "background: transparent;" in item_active_block
    assert "background: var(--accent-soft);" not in item_active_block
    assert "className = 'conv-menu-list hidden'" in html
    assert "textContent = '重命名'" in html
    assert "await renameConversation(item.session_id, item.title || '新对话');" in html
    assert "textContent = '删除对话'" in html
    assert "await deleteConversation(item.session_id);" in html
    assert "fetch(`/api/conversations/${encodeURIComponent(sessionId)}?user_id=${encodeURIComponent(user_id)}`" in html
